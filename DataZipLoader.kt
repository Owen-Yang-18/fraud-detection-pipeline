package your.package.name

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipInputStream

data class HeteroInputs(
    val x_app: Tensor, val x_sys: Tensor, val x_bnd: Tensor, val x_cmp: Tensor,
    val ei_app_sys: Tensor, val ew_app_sys: Tensor,
    val ei_sys_app: Tensor, val ew_sys_app: Tensor,
    val ei_app_bnd: Tensor, val ew_app_bnd: Tensor,
    val ei_bnd_app: Tensor, val ew_bnd_app: Tensor,
    val ei_app_cmp: Tensor, val ew_app_cmp: Tensor,
    val ei_cmp_app: Tensor, val ew_cmp_app: Tensor,
) {
    fun toEValues(): Array<EValue> = arrayOf(
        EValue.from(x_app), EValue.from(x_sys), EValue.from(x_bnd), EValue.from(x_cmp),
        EValue.from(ei_app_sys), EValue.from(ew_app_sys),
        EValue.from(ei_sys_app), EValue.from(ew_sys_app),
        EValue.from(ei_app_bnd), EValue.from(ew_app_bnd),
        EValue.from(ei_bnd_app), EValue.from(ew_bnd_app),
        EValue.from(ei_app_cmp), EValue.from(ew_app_cmp),
        EValue.from(ei_cmp_app), EValue.from(ew_cmp_app),
    )
}

data class Meta(
    val classNames: List<String>?,
    val appIds: List<String>?,
    val yApp: IntArray?,                        // ground truth 1..5, parallel to apps
    val labelMap: Map<Int, String>?             // optional manifest-provided mapping
)
data class Loaded(val inputs: HeteroInputs, val meta: Meta)

class DataZipLoader(private val ctx: Context) {

    fun load(uri: Uri): Loaded {
        val cr: ContentResolver = ctx.contentResolver

        // Read the ZIP into memory (simple for one-shot inference)
        val files = HashMap<String, ByteArray>()
        cr.openInputStream(uri)!!.use { input ->
            ZipInputStream(input).use { zis ->
                var entry = zis.nextEntry
                while (entry != null) {
                    val bos = ByteArrayOutputStream()
                    val buf = ByteArray(1 shl 16)
                    var r: Int
                    while (zis.read(buf).also { r = it } > 0) bos.write(buf, 0, r)
                    files[entry.name] = bos.toByteArray()
                    zis.closeEntry()
                    entry = zis.nextEntry
                }
            }
        }

        val manifest = JSONObject(String(files["manifest.json"] ?: error("manifest.json missing")))
        val tensors = manifest.getJSONArray("tensors")
        val classNames = manifest.optJSONArray("class_names")?.let { arr -> List(arr.length()) { i -> arr.getString(i) } }
        val appIds = manifest.optJSONArray("app_ids")?.let { arr -> List(arr.length()) { i -> arr.getString(i) } }
        val labelMap: Map<Int, String>? = manifest.optJSONObject("label_map")?.let { obj ->
            obj.keys().asSequence().associate { k -> k.toInt() to obj.getString(k) }
        }

        fun shapeOf(obj: JSONObject): LongArray {
            val arr = obj.getJSONArray("shape")
            return LongArray(arr.length()) { i -> arr.getLong(i) }
        }
        fun numel(shape: LongArray): Int {
            var n = 1L; for (s in shape) n *= s; return n.toInt()
        }
        fun loadFloat32(file: String, shape: LongArray): Tensor {
            val bytes = files[file] ?: error("Missing $file")
            val fa = FloatArray(numel(shape))
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(fa)
            return Tensor.fromBlob(fa, shape)
        }
        fun loadInt64Tensor(file: String, shape: LongArray): Tensor {
            val bytes = files[file] ?: error("Missing $file")
            val la = LongArray(numel(shape))
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer().get(la)
            return Tensor.fromBlob(la, shape)
        }
        fun loadInt64Vector(file: String, len: Int): IntArray {
            val bytes = files[file] ?: error("Missing $file")
            val la = LongArray(len)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer().get(la)
            return IntArray(len) { i -> la[i].toInt() }
        }

        val map = HashMap<String, Tensor>(16)
        for (i in 0 until tensors.length()) {
            val t = tensors.getJSONObject(i)
            val name = t.getString("name")
            val dtype = t.getString("dtype")
            val file = t.getString("file")
            val shape = shapeOf(t)
            val ten = when (dtype) {
                "float32" -> loadFloat32(file, shape)
                "int64"   -> loadInt64Tensor(file, shape)
                else -> error("Unsupported dtype $dtype for $name")
            }
            map[name] = ten
        }

        // Optional ground truth vector (int64, shape [N_app])
        val nApp = map["x_app"]!!.shape()[0].toInt()
        val yApp: IntArray? = if (files.containsKey("y_app.bin")) loadInt64Vector("y_app.bin", nApp) else null

        val inputs = HeteroInputs(
            map["x_app"]!!, map["x_sys"]!!, map["x_bnd"]!!, map["x_cmp"]!!,
            map["ei_app_sys"]!!, map["ew_app_sys"]!!,
            map["ei_sys_app"]!!, map["ew_sys_app"]!!,
            map["ei_app_bnd"]!!, map["ew_app_bnd"]!!,
            map["ei_bnd_app"]!!, map["ew_bnd_app"]!!,
            map["ei_app_cmp"]!!, map["ew_app_cmp"]!!,
            map["ei_cmp_app"]!!, map["ew_cmp_app"]!!,
        )
        return Loaded(inputs, Meta(classNames, appIds, yApp, labelMap))
    }
}
