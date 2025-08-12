package your.package.name

import android.os.Debug
import android.os.Process
import android.os.SystemClock
import kotlinx.coroutines.*
import kotlin.math.max
import kotlin.math.min

class PerfSampler(
    private val onUpdate: (Stats) -> Unit,
    private val periodMs: Long = 250L
) {
    data class Stats(
        val cpuPctAllCores: Double, // normalized to all cores
        val cpuPctOneCore: Double,  // normalized to a single core
        val pssMb: Double           // proportional set size, MB
    )

    private var job: Job? = null

    fun start() {
        stop()
        val scope = CoroutineScope(Dispatchers.Default)
        val cores = max(Runtime.getRuntime().availableProcessors(), 1)
        var lastCpuMs = Process.getElapsedCpuTime()            // process CPU ms (monotonic) :contentReference[oaicite:4]{index=4}
        var lastWallMs = SystemClock.elapsedRealtime()         // wall ms (monotonic) :contentReference[oaicite:5]{index=5}

        job = scope.launch {
            val mi = Debug.MemoryInfo()
            while (isActive) {
                delay(periodMs)
                val nowCpuMs = Process.getElapsedCpuTime()
                val nowWallMs = SystemClock.elapsedRealtime()

                val dCpu = max(nowCpuMs - lastCpuMs, 0L).toDouble()
                val dWall = max(nowWallMs - lastWallMs, 1L).toDouble()

                val cpuOneCore = 100.0 * dCpu / dWall
                val cpuAllCores = 100.0 * dCpu / (dWall * cores.toDouble())

                Debug.getMemoryInfo(mi)                        // fills MemoryInfo; PSS in kB :contentReference[oaicite:6]{index=6}
                val pssMb = mi.totalPss / 1024.0

                withContext(Dispatchers.Main) {
                    onUpdate(
                        Stats(
                            cpuPctAllCores = cpuAllCores.coerceIn(0.0, 100.0),
                            cpuPctOneCore  = min(cpuOneCore, 100.0),
                            pssMb = pssMb
                        )
                    )
                }
                lastCpuMs = nowCpuMs
                lastWallMs = nowWallMs
            }
        }
    }

    fun stop() { job?.cancel(); job = null }
}
