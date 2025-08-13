package your.package.name

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import your.package.name.databinding.ActivityMainBinding
import kotlinx.coroutines.*
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import java.io.File

class MainActivity : AppCompatActivity() {
  private lateinit var ui: ActivityMainBinding
  private val io = CoroutineScope(Dispatchers.IO)
  private var modelUri: Uri? = null
  private var dataUri: Uri? = null
  private var module: Module? = null

  private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
    uri ?: return@registerForActivityResult
    // Persist permission so we can reopen after reboot.
    contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
    modelUri = uri
    ui.txtModel.text = uri.toString()
  }

  private val pickData = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
    uri ?: return@registerForActivityResult
    contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
    dataUri = uri
    ui.txtData.text = uri.toString()
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    ui = ActivityMainBinding.inflate(layoutInflater)
    setContentView(ui.root)

    ui.btnSelectModel.setOnClickListener {
      pickModel.launch(arrayOf("*/*")) // let the user point to any .pte
    }
    ui.btnSelectData.setOnClickListener {
      pickData.launch(arrayOf("application/zip", "application/octet-stream", "*/*"))
    }
    ui.btnRun.setOnClickListener { runInference() }
  }

  private fun runInference() {
    val mUri = modelUri
    val dUri = dataUri
    if (mUri == null || dUri == null) {
      ui.txtOutput.text = "Pick both model and data first."
      return
    }

    ui.txtOutput.text = "Loading…"
    io.launch {
      try {
        // 1) Copy .pte to internal files (Module.load needs a filesystem path).
        val modelFile: File = File(filesDir, "model.pte").also {
          contentResolver.openInputStream(mUri)!!.use { it.copyTo(it@also.outputStream()) }
        }
        // 2) Load module (MMAP to reduce peak memory).
        module?.destroy()
        module = Module.load(modelFile.absolutePath, Module.LOAD_MODE_MMAP)

        // 3) Read data ZIP -> tensors + metadata (class names, app IDs).
        val loaded = DataZipLoader(this@MainActivity).load(dUri)

        // 4) Run forward. Order must match your Python model signature.
        val evalues: Array<EValue> = loaded.inputs.toEValues()
        val out = module!!.forward(evalues)[0].toTensor()
        val logits = out.getDataAsFloatArray()
        val shape = out.shape() // [N_app, num_classes]

        // 5) Post-process: argmax per app row, map to label + app_id.
        val nApp = shape[0].toInt()
        val nCls = shape[1].toInt()
        val sb = StringBuilder()
        sb.append("Output: [${shape.joinToString("x")}]\n")
        for (i in 0 until nApp) {
          var best = 0
          var bestVal = Float.NEGATIVE_INFINITY
          val base = i * nCls
          for (c in 0 until nCls) {
            val v = logits[base + c]
            if (v > bestVal) { bestVal = v; best = c }
          }
          val appId = loaded.meta.appIds?.getOrNull(i) ?: "app_$i"
          val label = loaded.meta.classNames?.getOrNull(best) ?: "class_$best"
          sb.append("$appId -> $label (logit=${"%.4f".format(bestVal)})\n")
        }

        withContext(Dispatchers.Main) { ui.txtOutput.text = sb.toString() }
      } catch (t: Throwable) {
        withContext(Dispatchers.Main) { ui.txtOutput.text = "Error: ${t.message}" }
      }
    }
  }

  override fun onDestroy() {
    module?.destroy()
    super.onDestroy()
  }
}



package your.package.name

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import your.package.name.databinding.ActivityMainBinding
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var ui: ActivityMainBinding
    private val io = CoroutineScope(Dispatchers.IO)

    private var modelUri: Uri? = null
    private var dataUri: Uri? = null
    private var module: Module? = null

    companion object {
        private const val KEY_MODEL_URI = "key_model_uri"
        private const val KEY_DATA_URI  = "key_data_uri"
        private const val PREFS = "picker_prefs"
    }

    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        try {
            // Persist long-term read access for this Uri (SAF).
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        } catch (_: SecurityException) { /* some providers may not support persistable grants */ }
        modelUri = uri
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putString(KEY_MODEL_URI, uri.toString())
            .apply()
        ui.txtModel.text = uri.toString()
    }

    private val pickData = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        try {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        } catch (_: SecurityException) { }
        dataUri = uri
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putString(KEY_DATA_URI, uri.toString())
            .apply()
        ui.txtData.text = uri.toString()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ui = ActivityMainBinding.inflate(layoutInflater)
        setContentView(ui.root)

        // 1) Restore from a prior recreation if present
        if (savedInstanceState != null) {
            modelUri = savedInstanceState.getParcelable(KEY_MODEL_URI)
            dataUri  = savedInstanceState.getParcelable(KEY_DATA_URI)
        }

        // 2) Fallback to SharedPreferences (process-death resilience)
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        if (modelUri == null) prefs.getString(KEY_MODEL_URI, null)?.let { modelUri = Uri.parse(it) }
        if (dataUri  == null) prefs.getString(KEY_DATA_URI,  null)?.let { dataUri  = Uri.parse(it) }

        ui.txtModel.text = modelUri?.toString() ?: "No model selected"
        ui.txtData.text  = dataUri?.toString()  ?: "No data selected"

        ui.btnSelectModel.setOnClickListener {
            pickModel.launch(arrayOf("*/*")) // pick any .pte
        }
        ui.btnSelectData.setOnClickListener {
            // ZIP with manifest; allow generic if provider does not set type
            pickData.launch(arrayOf("application/zip", "application/octet-stream", "*/*"))
        }
        ui.btnRun.setOnClickListener { runInference() }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        modelUri?.let { outState.putParcelable(KEY_MODEL_URI, it) }
        dataUri ?.let { outState.putParcelable(KEY_DATA_URI,  it) }
        // (SharedPreferences already updated on selection)
    }

    private fun runInference() {
        val mUri = modelUri
        val dUri = dataUri
        if (mUri == null || dUri == null) {
            ui.txtOutput.text = "Pick both model and data first."
            return
        }

        ui.txtOutput.text = "Loading…"
        io.launch {
            try {
                // Copy model to an actual file path (ExecuTorch loads from filesystem).
                val modelFile = File(filesDir, "model.pte")
                contentResolver.openInputStream(mUri)!!.use { input ->
                    FileOutputStream(modelFile, false).use { out -> input.copyTo(out) }
                }

                module?.destroy()
                module = Module.load(modelFile.absolutePath, Module.LOAD_MODE_MMAP)

                // Load data ZIP -> tensors + metadata
                val loaded = DataZipLoader(this@MainActivity).load(dUri)
                val evalues: Array<EValue> = loaded.inputs.toEValues()

                // ExecuTorch AARs differ; 'execute' works across versions.
                val outs = module!!.execute("forward", evalues)
                val outTensor = outs[0].toTensor()
                val logits = outTensor.getDataAsFloatArray()
                val shape = outTensor.shape() // [N_app, num_classes]

                val nApp = shape[0].toInt()
                val nCls = shape[1].toInt()
                val sb = StringBuilder().apply { append("Output: [${shape.joinToString("x")}]\n") }
                for (i in 0 until nApp) {
                    var best = 0; var bestVal = Float.NEGATIVE_INFINITY
                    val base = i * nCls
                    for (c in 0 until nCls) {
                        val v = logits[base + c]
                        if (v > bestVal) { bestVal = v; best = c }
                    }
                    val appId = loaded.meta.appIds?.getOrNull(i) ?: "app_$i"
                    val label = loaded.meta.classNames?.getOrNull(best) ?: "class_$best"
                    sb.append("$appId -> $label (logit=${"%.4f".format(bestVal)})\n")
                }

                withContext(Dispatchers.Main) { ui.txtOutput.text = sb.toString() }
            } catch (t: Throwable) {
                withContext(Dispatchers.Main) { ui.txtOutput.text = "Error: ${t.message}" }
            }
        }
    }

    override fun onDestroy() {
        module?.destroy()
        super.onDestroy()
    }
}


package your.package.name

import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import your.package.name.databinding.ActivityMainBinding
import java.io.File
import java.io.FileOutputStream
import java.text.DecimalFormat
import java.text.DecimalFormatSymbols
import java.util.Locale
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    private lateinit var ui: ActivityMainBinding
    private val io = CoroutineScope(Dispatchers.IO)

    private var modelUri: Uri? = null
    private var dataUri: Uri? = null
    private var module: Module? = null
    private var perf: PerfSampler? = null

    // Simple, locale-stable number formatters (avoid IllegalFormatPrecisionException)
    private val dfs1 = DecimalFormat("0.0",   DecimalFormatSymbols(Locale.US))
    private val dfs2 = DecimalFormat("0.00",  DecimalFormatSymbols(Locale.US))
    private fun f1(x: Double) = dfs1.format(x)
    private fun f2(x: Double) = dfs2.format(x)

    companion object {
        private const val KEY_MODEL_URI = "key_model_uri"
        private const val KEY_DATA_URI  = "key_data_uri"
        private const val PREFS = "picker_prefs"
    }

    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        try { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) } catch (_: SecurityException) {}
        modelUri = uri
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KEY_MODEL_URI, uri.toString()).apply()
        ui.txtModel.text = uri.toString()
    }

    private val pickData = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        try { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) } catch (_: SecurityException) {}
        dataUri = uri
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KEY_DATA_URI, uri.toString()).apply()
        ui.txtData.text = uri.toString()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ui = ActivityMainBinding.inflate(layoutInflater)
        setContentView(ui.root)

        if (savedInstanceState != null) {
            modelUri = savedInstanceState.getParcelable(KEY_MODEL_URI)
            dataUri  = savedInstanceState.getParcelable(KEY_DATA_URI)
        }
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        if (modelUri == null) prefs.getString(KEY_MODEL_URI, null)?.let { modelUri = Uri.parse(it) }
        if (dataUri  == null) prefs.getString(KEY_DATA_URI,  null)?.let { dataUri  = Uri.parse(it) }

        ui.txtModel.text = modelUri?.toString() ?: "No model selected"
        ui.txtData.text  = dataUri?.toString()  ?: "No data selected"

        ui.btnSelectModel.setOnClickListener { pickModel.launch(arrayOf("*/*")) }
        ui.btnSelectData.setOnClickListener { pickData.launch(arrayOf("application/zip", "application/octet-stream", "*/*")) }
        ui.btnRun.setOnClickListener { runInference() }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        modelUri?.let { outState.putParcelable(KEY_MODEL_URI, it) }
        dataUri ?.let { outState.putParcelable(KEY_DATA_URI,  it) }
    }

    private fun runInference() {
        val mUri = modelUri
        val dUri = dataUri
        if (mUri == null || dUri == null) {
            ui.txtOutput.text = "Pick both model and data first."
            return
        }

        ui.txtOutput.text = "Loading…"
        perf?.stop()
        perf = PerfSampler(onUpdate = { s ->
            runOnUiThread {
                ui.txtStats.text =
                    "CPU (all cores): ${f1(s.cpuPctAllCores)}%   " +
                    "CPU (1 core): ${f1(s.cpuPctOneCore)}%   " +
                    "PSS: ${f1(s.pssMb)} MB"
            }
        }).also { it.start() }

        io.launch {
            try {
                // Copy model to internal storage and load
                val modelFile = File(filesDir, "model.pte")
                contentResolver.openInputStream(mUri)!!.use { input ->
                    FileOutputStream(modelFile, false).use { out -> input.copyTo(out) }
                }
                module?.destroy()
                module = Module.load(modelFile.absolutePath, Module.LOAD_MODE_MMAP)

                // Load tensors + metadata (now includes y_app and optional label_map)
                val loaded = DataZipLoader(this@MainActivity).load(dUri)
                val evalues: Array<EValue> = loaded.inputs.toEValues()

                // Inference timing
                val t0 = SystemClock.elapsedRealtimeNanos()
                val outs = module!!.execute("forward", evalues)
                val t1 = SystemClock.elapsedRealtimeNanos()
                val totalMs = (t1 - t0) / 1e6
                val outTensor = outs[0].toTensor()
                val shape = outTensor.shape() // [N_app, num_classes]
                val logits = outTensor.getDataAsFloatArray()
                val nApp = shape[0].toInt()
                val nCls = shape[1].toInt()
                val perSampleMs = totalMs / max(nApp, 1)

                // Label mapping: prefer manifest label_map; else default 1..5 mapping
                val labelMap = loaded.meta.labelMap ?: mapOf(
                    1 to "adware", 2 to "banking", 3 to "sms", 4 to "riskware", 5 to "benign"
                )
                val yTrue: IntArray? = loaded.meta.yApp // expected values 1..5 (same length as apps)

                // Build colored output
                val span = SpannableStringBuilder()
                for (i in 0 until nApp) {
                    // argmax on row i
                    var best = 0; var bestVal = Float.NEGATIVE_INFINITY
                    val base = i * nCls
                    for (c in 0 until nCls) {
                        val v = logits[base + c]
                        if (v > bestVal) { bestVal = v; best = c }
                    }
                    val predIdx1 = best + 1 // model is 0-based; ground truth is 1..5
                    val predLabel = labelMap[predIdx1] ?: "class_$predIdx1"
                    val appId = loaded.meta.appIds?.getOrNull(i) ?: "app_$i"

                    val trueIdx1 = yTrue?.getOrNull(i)
                    val trueLabel = trueIdx1?.let { labelMap[it] ?: "class_$it" }

                    val ok = trueIdx1 != null && trueIdx1 == predIdx1
                    val color = if (ok) Color.parseColor("#1976D2") else Color.parseColor("#D32F2F")

                    val line = if (trueLabel != null)
                        "$appId -> pred: $predLabel | true: $trueLabel\n"
                    else
                        "$appId -> pred: $predLabel\n"

                    val start = span.length
                    span.append(line)
                    span.setSpan(ForegroundColorSpan(color), start, span.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                }

                withContext(Dispatchers.Main) {
                    ui.txtOutput.text =
                        "Total: ${f2(totalMs)} ms   Per-sample: ${f2(perSampleMs)} ms\n\n"
                    ui.txtOutput.append(span)
                }
            } catch (t: Throwable) {
                withContext(Dispatchers.Main) { ui.txtOutput.text = "Error: ${t.message}" }
            } finally {
                withContext(Dispatchers.Main) { perf?.stop() }
            }
        }
    }

    override fun onStop() { super.onStop(); perf?.stop() }
    override fun onDestroy() { perf?.stop(); module?.destroy(); super.onDestroy() }
}


package your.package.name

import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import your.package.name.databinding.ActivityMainBinding
import java.io.File
import java.io.FileOutputStream
import java.text.DecimalFormat
import java.text.DecimalFormatSymbols
import java.util.Locale
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    private lateinit var ui: ActivityMainBinding
    private val io = CoroutineScope(Dispatchers.IO)

    private var modelUri: Uri? = null
    private var dataUri: Uri? = null
    private var module: Module? = null
    private var perf: PerfSampler? = null

    // Locale-stable formatters
    private val dfs1 = DecimalFormat("0.0",    DecimalFormatSymbols(Locale.US))
    private val dfs2 = DecimalFormat("0.00",   DecimalFormatSymbols(Locale.US))
    private val dfs4 = DecimalFormat("0.0000", DecimalFormatSymbols(Locale.US))
    private fun f1(x: Double) = dfs1.format(x)
    private fun f2(x: Double) = dfs2.format(x)
    private fun f4(x: Double) = dfs4.format(x)
    private fun pct4(x: Double) = f4(x * 100.0) + "%"

    companion object {
        private const val KEY_MODEL_URI = "key_model_uri"
        private const val KEY_DATA_URI  = "key_data_uri"
        private const val PREFS = "picker_prefs"
    }

    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        try { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) } catch (_: SecurityException) {}
        modelUri = uri
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KEY_MODEL_URI, uri.toString()).apply()
        ui.txtModel.text = uri.toString()
    }

    private val pickData = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        try { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) } catch (_: SecurityException) {}
        dataUri = uri
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KEY_DATA_URI, uri.toString()).apply()
        ui.txtData.text = uri.toString()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ui = ActivityMainBinding.inflate(layoutInflater)
        setContentView(ui.root)

        if (savedInstanceState != null) {
            modelUri = savedInstanceState.getParcelable(KEY_MODEL_URI)
            dataUri  = savedInstanceState.getParcelable(KEY_DATA_URI)
        }
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        if (modelUri == null) prefs.getString(KEY_MODEL_URI, null)?.let { modelUri = Uri.parse(it) }
        if (dataUri  == null) prefs.getString(KEY_DATA_URI,  null)?.let { dataUri  = Uri.parse(it) }

        ui.txtModel.text = modelUri?.toString() ?: "No model selected"
        ui.txtData.text  = dataUri?.toString()  ?: "No data selected"

        ui.btnSelectModel.setOnClickListener { pickModel.launch(arrayOf("*/*")) }
        ui.btnSelectData.setOnClickListener { pickData.launch(arrayOf("application/zip", "application/octet-stream", "*/*")) }
        ui.btnRun.setOnClickListener { runInference() }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        modelUri?.let { outState.putParcelable(KEY_MODEL_URI, it) }
        dataUri ?.let { outState.putParcelable(KEY_DATA_URI,  it) }
    }

    private fun runInference() {
        val mUri = modelUri
        val dUri = dataUri
        if (mUri == null || dUri == null) {
            ui.txtOutput.text = "Pick both model and data first."
            return
        }

        ui.txtOutput.text = "Loading…"
        perf?.stop()
        perf = PerfSampler(onUpdate = { s ->
            runOnUiThread {
                ui.txtStats.text =
                    "CPU (all cores): ${f1(s.cpuPctAllCores)}%   " +
                    "CPU (1 core): ${f1(s.cpuPctOneCore)}%   " +
                    "PSS: ${f1(s.pssMb)} MB"
            }
        }).also { it.start() }

        io.launch {
            try {
                // Copy and load model
                val modelFile = File(filesDir, "model.pte")
                contentResolver.openInputStream(mUri)!!.use { input ->
                    FileOutputStream(modelFile, false).use { out -> input.copyTo(out) }
                }
                module?.destroy()
                module = Module.load(modelFile.absolutePath, Module.LOAD_MODE_MMAP)

                // Load data (includes optional y_app and label_map)
                val loaded = DataZipLoader(this@MainActivity).load(dUri)
                val evalues: Array<EValue> = loaded.inputs.toEValues()

                // Inference timing
                val t0 = SystemClock.elapsedRealtimeNanos()
                val outs = module!!.execute("forward", evalues)
                val t1 = SystemClock.elapsedRealtimeNanos()
                val totalMs = (t1 - t0) / 1e6
                val outTensor = outs[0].toTensor()
                val shape = outTensor.shape() // [N_app, num_classes]
                val logits = outTensor.getDataAsFloatArray()

                val nApp = shape[0].toInt()
                val nCls = shape[1].toInt()
                val perSampleMs = totalMs / max(nApp, 1)

                // Labels
                val labelMap = loaded.meta.labelMap ?: mapOf(
                    1 to "adware", 2 to "banking", 3 to "sms", 4 to "riskware", 5 to "benign"
                )
                val yTrue: IntArray? = loaded.meta.yApp  // 1..5

                // Build yPred (1..nCls) + colored per-app lines
                val span = SpannableStringBuilder()
                val yPredIdx1 = IntArray(nApp)
                for (i in 0 until nApp) {
                    // argmax
                    var best = 0; var bestVal = Float.NEGATIVE_INFINITY
                    val base = i * nCls
                    for (c in 0 until nCls) {
                        val v = logits[base + c]
                        if (v > bestVal) { bestVal = v; best = c }
                    }
                    val predIdx1 = best + 1
                    yPredIdx1[i] = predIdx1

                    val predLabel = labelMap[predIdx1] ?: "class_$predIdx1"
                    val rawId = loaded.meta.appIds?.getOrNull(i) ?: "app_$i"
                    val shownId = AppIdHasher.code(rawId)

                    val trueIdx1 = yTrue?.getOrNull(i)
                    val trueLabel = trueIdx1?.let { labelMap[it] ?: "class_$it" }

                    val ok = trueIdx1 != null && trueIdx1 == predIdx1
                    val color = if (ok) Color.parseColor("#1976D2") else Color.parseColor("#D32F2F")

                    val line = if (trueLabel != null)
                        "$shownId -> pred: $predLabel | true: $trueLabel\n"
                    else
                        "$shownId -> pred: $predLabel\n"

                    val start = span.length
                    span.append(line)
                    span.setSpan(ForegroundColorSpan(color), start, span.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                }

                // Compute metrics (if we have y_true)
                val metricsText: String = if (yTrue != null && yTrue.size == nApp) {
                    val m = computeMacroMetrics(yTrue, yPredIdx1, nCls)
                    buildString {
                        append("Accuracy: ${pct4(m.acc)}\n")
                        append("F1 (macro): ${pct4(m.f1)}\n")
                        append("Precision (macro): ${pct4(m.precision)}\n")
                        append("Recall (macro): ${pct4(m.recall)}\n")
                        append("TPR (macro): ${pct4(m.tpr)}\n")
                        append("TNR (macro): ${pct4(m.tnr)}\n")
                        append("FPR (macro): ${pct4(m.fpr)}\n")
                        append("FNR (macro): ${pct4(m.fnr)}\n")
                    }
                } else {
                    "No ground-truth labels found (y_app.bin)."
                }

                withContext(Dispatchers.Main) {
                    // 1) performance
                    ui.txtOutput.text =
                        "Total: ${f2(totalMs)} ms   Per-sample: ${f2(perSampleMs)} ms\n\n"
                    // 2) metrics
                    ui.txtOutput.append(metricsText + "\n\n")
                    // 3) per-app lines
                    ui.txtOutput.append(span)
                }
            } catch (t: Throwable) {
                withContext(Dispatchers.Main) { ui.txtOutput.text = "Error: ${t.message}" }
            } finally {
                withContext(Dispatchers.Main) { perf?.stop() }
            }
        }
    }

    override fun onStop() { super.onStop(); perf?.stop() }
    override fun onDestroy() { perf?.stop(); module?.destroy(); super.onDestroy() }

    // -------------------- metrics --------------------

    data class Metrics(
        val acc: Double,
        val precision: Double, // macro
        val recall: Double,    // macro
        val f1: Double,        // macro
        val tpr: Double,       // macro (same as recall, but we compute explicitly)
        val tnr: Double,       // macro
        val fpr: Double,       // macro
        val fnr: Double        // macro
    )

    /**
     * Macro-averaged metrics using one-vs-rest for K classes (labels 1..K).
     * For each class c:
     *   TP = pred=c & true=c
     *   FP = pred=c & true!=c
     *   FN = pred!=c & true=c
     *   TN = N - TP - FP - FN
     * We average only over classes with a valid denominator for the metric.
     */
    private fun computeMacroMetrics(yTrue1: IntArray, yPred1: IntArray, k: Int): Metrics {
        val n = yTrue1.size
        var correct = 0L

        val pos = LongArray(k + 1)     // count of true==c
        val pred = LongArray(k + 1)    // count of pred==c
        val tp = LongArray(k + 1)

        for (i in 0 until n) {
            val t = yTrue1[i]
            val p = yPred1[i]
            if (t == p) correct++
            pos[t]++
            pred[p]++
            if (t == p) tp[t]++
        }
        val N = n.toLong()

        var sumPrec = 0.0; var cntPrec = 0
        var sumRec  = 0.0; var cntRec  = 0
        var sumF1   = 0.0; var cntF1   = 0
        var sumTPR  = 0.0; var cntTPR  = 0
        var sumTNR  = 0.0; var cntTNR  = 0
        var sumFPR  = 0.0; var cntFPR  = 0
        var sumFNR  = 0.0; var cntFNR  = 0

        for (c in 1..k) {
            val tpC = tp[c]
            val fpC = pred[c] - tpC
            val fnC = pos[c] - tpC
            val tnC = N - tpC - fpC - fnC

            // precision_c
            val denomP = tpC + fpC
            if (denomP > 0) { sumPrec += tpC.toDouble() / denomP; cntPrec++ }

            // recall_c (TPR)
            val denomR = tpC + fnC
            if (denomR > 0) { sumRec += tpC.toDouble() / denomR; cntRec++ }

            // F1_c
            if (denomP > 0 && denomR > 0) {
                val pC = tpC.toDouble() / denomP
                val rC = tpC.toDouble() / denomR
                if (pC + rC > 0.0) { sumF1 += 2.0 * pC * rC / (pC + rC); cntF1++ }
            }

            // TPR (= recall)
            if (denomR > 0) { sumTPR += tpC.toDouble() / denomR; cntTPR++ }

            // TNR
            val denomTNR = tnC + fpC
            if (denomTNR > 0) { sumTNR += tnC.toDouble() / denomTNR; cntTNR++ }

            // FPR
            val denomFPR = fpC + tnC
            if (denomFPR > 0) { sumFPR += fpC.toDouble() / denomFPR; cntFPR++ }

            // FNR
            val denomFNR = fnC + tpC
            if (denomFNR > 0) { sumFNR += fnC.toDouble() / denomFNR; cntFNR++ }
        }

        val acc = correct.toDouble() / n.toDouble()
        fun avg(sum: Double, cnt: Int) = if (cnt > 0) sum / cnt else 0.0

        return Metrics(
            acc = acc,
            precision = avg(sumPrec, cntPrec),
            recall    = avg(sumRec,  cntRec),
            f1        = avg(sumF1,   cntF1),
            tpr       = avg(sumTPR,  cntTPR),
            tnr       = avg(sumTNR,  cntTNR),
            fpr       = avg(sumFPR,  cntFPR),
            fnr       = avg(sumFNR,  cntFNR),
        )
    }
}
