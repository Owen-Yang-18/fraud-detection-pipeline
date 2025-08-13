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


// app/src/main/java/your/package/name/PerfSampler.kt
package your.package.name

import android.os.Debug
import android.os.Process
import android.os.SystemClock
import kotlinx.coroutines.*
import kotlin.math.max
import kotlin.math.min

class PerfSampler(
    private val periodMs: Long = 250L,
    private val onUpdate: (Stats) -> Unit,
    private val onSummary: ((Summary) -> Unit)? = null
) {
    data class Stats(
        val cpuPctAllCores: Double,   // 0..100 normalized to all cores
        val cpuPctOneCore: Double,    // 0..100 on one core
        val pssMb: Double,            // current PSS (MB)
        val ussMb: Double,            // current USS (MB ~ privateClean+privateDirty)
        val dPssMb: Double,           // PSS - baseline (MB)
        val dUssMb: Double,           // USS - baseline (MB)
        val peakPssMb: Double,        // peak PSS since start (MB)
        val peakUssMb: Double,        // peak USS since start (MB)
        val elapsedMs: Double         // wall time since start
    )

    data class Summary(
        val durationMs: Double,
        val basePssMb: Double, val endPssMb: Double, val dPssMb: Double, val peakPssMb: Double,
        val baseUssMb: Double, val endUssMb: Double, val dUssMb: Double, val peakUssMb: Double
    )

    private var job: Job? = null

    private var basePssKb = -1
    private var baseUssKb = -1
    private var peakPssKb = 0
    private var peakUssKb = 0

    private var lastCpuMs = 0L
    private var lastWallMs = 0L
    private var startNs = 0L

    fun start() {
        stop()
        val scope = CoroutineScope(Dispatchers.Default)
        val cores = max(Runtime.getRuntime().availableProcessors(), 1)

        lastCpuMs = Process.getElapsedCpuTime()
        lastWallMs = SystemClock.elapsedRealtime()
        startNs = SystemClock.elapsedRealtimeNanos()

        basePssKb = -1
        baseUssKb = -1
        peakPssKb = 0
        peakUssKb = 0

        job = scope.launch {
            val mi = Debug.MemoryInfo()
            while (isActive) {
                delay(periodMs)

                // CPU deltas
                val nowCpuMs = Process.getElapsedCpuTime()
                val nowWallMs = SystemClock.elapsedRealtime()
                val dCpu = max(nowCpuMs - lastCpuMs, 0L).toDouble()
                val dWall = max(nowWallMs - lastWallMs, 1L).toDouble()
                val cpuOneCore = 100.0 * dCpu / dWall
                val cpuAllCores = 100.0 * dCpu / (dWall * cores.toDouble())
                lastCpuMs = nowCpuMs
                lastWallMs = nowWallMs

                // Memory (kB)
                Debug.getMemoryInfo(mi)
                val pssKb = mi.totalPss
                val ussKb = mi.totalPrivateClean + mi.totalPrivateDirty  // ~ USS

                if (basePssKb < 0) basePssKb = pssKb
                if (baseUssKb < 0) baseUssKb = ussKb
                if (pssKb > peakPssKb) peakPssKb = pssKb
                if (ussKb > peakUssKb) peakUssKb = ussKb

                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1e6

                onUpdate(
                    Stats(
                        cpuPctAllCores = cpuAllCores.coerceIn(0.0, 100.0),
                        cpuPctOneCore  = min(cpuOneCore, 100.0),
                        pssMb = pssKb / 1024.0,
                        ussMb = ussKb / 1024.0,
                        dPssMb = (pssKb - basePssKb) / 1024.0,
                        dUssMb = (ussKb - baseUssKb) / 1024.0,
                        peakPssMb = peakPssKb / 1024.0,
                        peakUssMb = peakUssKb / 1024.0,
                        elapsedMs = elapsedMs
                    )
                )
            }
        }
    }

    fun stop() {
        val j = job
        job = null
        j?.cancel()

        // Final summary (fresh read)
        val mi = Debug.MemoryInfo()
        Debug.getMemoryInfo(mi)
        val endPssKb = mi.totalPss
        val endUssKb = mi.totalPrivateClean + mi.totalPrivateDirty
        val basePss = if (basePssKb >= 0) basePssKb else endPssKb
        val baseUss = if (baseUssKb >= 0) baseUssKb else endUssKb
        val durMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1e6

        onSummary?.invoke(
            Summary(
                durationMs = durMs,
                basePssMb = basePss / 1024.0, endPssMb = endPssKb / 1024.0,
                dPssMb = (endPssKb - basePss) / 1024.0, peakPssMb = peakPssKb / 1024.0,
                baseUssMb = baseUss / 1024.0, endUssMb = endUssKb / 1024.0,
                dUssMb = (endUssKb - baseUss) / 1024.0, peakUssMb = peakUssKb / 1024.0
            )
        )
    }
}

