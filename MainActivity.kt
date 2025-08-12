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
