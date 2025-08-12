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

// -------- Data classes that match your Python model inputs --------
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

data class Meta(val classNames: List<String>?, val appIds: List<String>?)
data class Loaded(val inputs: HeteroInputs, val meta: Meta)

class DataZipLoader(private val ctx: Context) {
  fun load(uri: Uri): Loaded {
    val cr: ContentResolver = ctx.contentResolver

    // 1) Read entire ZIP into a map: name -> bytes
    val files = HashMap<String, ByteArray>()
    cr.openInputStream(uri)!!.use { input ->
      ZipInputStream(input).use { zis ->
        var entry = zis.nextEntry
        while (entry != null) {
          val name = entry.name
          val bos = ByteArrayOutputStream()
          val buf = ByteArray(1 shl 16)
          var r: Int
          while (zis.read(buf).also { r = it } > 0) bos.write(buf, 0, r)
          files[name] = bos.toByteArray()
          zis.closeEntry()
          entry = zis.nextEntry
        }
      }
    }

    // 2) Parse manifest.json
    val manifest = JSONObject(String(files["manifest.json"] ?: error("manifest.json missing")))
    val tensors = manifest.getJSONArray("tensors")
    val classNames = manifest.optJSONArray("class_names")?.let { arr ->
      List(arr.length()) { i -> arr.getString(i) }
    }
    val appIds = manifest.optJSONArray("app_ids")?.let { arr ->
      List(arr.length()) { i -> arr.getString(i) }
    }

    fun shapeOf(obj: JSONObject): LongArray {
      val arr = obj.getJSONArray("shape")
      return LongArray(arr.length()) { i -> arr.getLong(i) }
    }
    fun loadFloat32(file: String, shape: LongArray): Tensor {
      val bytes = files[file] ?: error("Missing $file")
      val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
      val fa = FloatArray((Tensor.numel(shape)).toInt())
      bb.asFloatBuffer().get(fa)
      return Tensor.fromBlob(fa, shape)
    }
    fun loadInt64(file: String, shape: LongArray): Tensor {
      val bytes = files[file] ?: error("Missing $file")
      val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
      val la = LongArray((Tensor.numel(shape)).toInt())
      bb.asLongBuffer().get(la)
      return Tensor.fromBlob(la, shape)
    }

    // 3) Build a map name->Tensor
    val map = HashMap<String, Tensor>(16)
    for (i in 0 until tensors.length()) {
      val t = tensors.getJSONObject(i)
      val name = t.getString("name")
      val dtype = t.getString("dtype")
      val file = t.getString("file")
      val shape = shapeOf(t)
      val ten = when (dtype) {
        "float32" -> loadFloat32(file, shape)
        "int64"   -> loadInt64(file, shape)
        else -> error("Unsupported dtype $dtype")
      }
      map[name] = ten
    }

    // 4) Package in the exact order your model expects
    val inputs = HeteroInputs(
      map["x_app"]!!, map["x_sys"]!!, map["x_bnd"]!!, map["x_cmp"]!!,
      map["ei_app_sys"]!!, map["ew_app_sys"]!!,
      map["ei_sys_app"]!!, map["ew_sys_app"]!!,
      map["ei_app_bnd"]!!, map["ew_app_bnd"]!!,
      map["ei_bnd_app"]!!, map["ew_bnd_app"]!!,
      map["ei_app_cmp"]!!, map["ew_app_cmp"]!!,
      map["ei_cmp_app"]!!, map["ew_cmp_app"]!!,
    )
    return Loaded(inputs, Meta(classNames, appIds))
  }
}
