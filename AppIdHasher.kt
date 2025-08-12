package your.package.name

import java.math.BigInteger
import java.security.MessageDigest
import java.util.Locale

object AppIdHasher {
    /**
     * Deterministic 8-char code in [0-9A-Z] from an appId string.
     * SHA-256 -> positive BigInteger -> base36 -> upper -> first 8 chars (left-pad '0' if short).
     */
    fun code(appId: String): String {
        val bytes = MessageDigest.getInstance("SHA-256")
            .digest(appId.toByteArray(Charsets.UTF_8))          // SHA-256. :contentReference[oaicite:1]{index=1}
        val bi = BigInteger(1, bytes)                           // positive
        val base36 = bi.toString(36).uppercase(Locale.US)       // base-36 digits 0-9, A-Z. :contentReference[oaicite:2]{index=2}
        val eight = if (base36.length >= 8) base36.substring(0, 8)
                    else base36.padStart(8, '0')
        return eight
    }
}
