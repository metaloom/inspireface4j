package io.metaloom.facedetect4j.yunet.onnx;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Makes the GPU work without asking the caller to set up an environment first.
 *
 * <p>
 * There are two reasons a JVM with a perfectly good CUDA installation still cannot run ONNX
 * Runtime, and this class removes both from the caller's plate.
 *
 * <h2>1. The Maven artifact is a CUDA 12 build</h2>
 * {@code com.microsoft.onnxruntime:onnxruntime_gpu:1.28.0} links {@code libcudart.so.12}, while the
 * PyPI wheel of the <i>same version</i> links {@code libcudart.so.13}. onnxruntime.ai's "1.27+ is
 * CUDA 13" table describes the PyPI and NuGet packages, not the Java one. On a CUDA 13 host the
 * jar's provider therefore cannot load — and if some unrelated package left an old
 * {@code libcudart.so.12} lying around (Debian ships 12.4 as {@code libcudart12}) it loads far
 * enough to fail with {@code undefined symbol: cudaLibraryGetKernel}, which names neither CUDA nor
 * the version as the problem.
 *
 * <p>
 * {@code setup-cuda.sh} puts the matching natives on disk; this class points ONNX Runtime at them
 * by setting {@code onnxruntime.native.path} <i>before</i> the runtime's own class initialiser
 * reads it.
 *
 * <h2>2. cuDNN is dlopen'd, not linked</h2>
 * cuDNN is absent from the provider's {@code NEEDED} list because ONNX Runtime opens it lazily at
 * the first Conv node — so a missing one surfaces after the provider has attached and the session
 * has been created, looking like a model problem. No distribution packages it either: NVIDIA's
 * {@code debian13} CUDA repo carries zero cuDNN packages, and {@code compute/cudnn/repos} has no
 * Debian tree at all, so the tarball is the only route and it lands somewhere the loader does not
 * search.
 *
 * <p>
 * The usual answer is {@code LD_LIBRARY_PATH}, which has to be exported before the process starts
 * and therefore infects every launcher — shell, IDE, surefire fork, downstream application.
 * {@link System#load(String)} on the absolute path avoids that: an object loaded that way is
 * registered under its {@code SONAME}, so ONNX Runtime's later {@code dlopen} finds it already
 * resident and never consults the search path. Same trick covers a project-local CUDA runtime.
 *
 * <p>
 * Load order is not computed — cuDNN 9 is a dozen libraries with a dependency graph among them.
 * Instead the loads are retried to a fixpoint: a library whose dependency has not been loaded yet
 * fails, and succeeds on a later pass once it has.
 *
 * <h2>When this does nothing</h2>
 * On a host where ONNX Runtime already works — CUDA 12 plus cuDNN on the default loader path — no
 * provisioning directory exists, nothing is found, and the stock behaviour is untouched. An
 * explicitly set {@code onnxruntime.native.path} is never overwritten.
 */
final class CudaNatives {

	private static final Logger log = LoggerFactory.getLogger(CudaNatives.class);

	/** ONNX Runtime's own switch for loading natives from a directory instead of the jar. */
	private static final String ORT_NATIVE_PATH = "onnxruntime.native.path";

	/** Overrides the search below, for a provisioning directory kept somewhere else entirely. */
	private static final String DIR_PROPERTY = "facedetect4j.cuda.dir";
	private static final String DIR_ENV = "FACEDETECT4J_CUDA_DIR";

	/** What {@code setup-cuda.sh} creates, looked for in the working directory and its parents. */
	private static final String LOCAL_DIR = ".cuda";
	private static final int SEARCH_DEPTH = 6;

	/**
	 * A versioned shared library, i.e. a {@code SONAME}. Matching this rather than {@code *.so}
	 * skips the unversioned development symlinks, which point at the same files and would be loaded
	 * twice under a name nothing ever asks for.
	 */
	private static final String SONAME = "lib.+\\.so\\.\\d+";

	private static boolean done;

	private CudaNatives() {
	}

	/**
	 * Idempotent, and safe to call when no provisioning exists.
	 *
	 * <p>
	 * Must run before any ONNX Runtime class initialises, because that is when
	 * {@code onnxruntime.native.path} is read. {@link OrtRuntime} is the only door into ONNX Runtime
	 * in this module and calls this from its static initialiser, which is what guarantees the
	 * ordering.
	 */
	static synchronized void bootstrap() {
		if (done) {
			return;
		}
		done = true;
		Optional<Path> root = locate();
		if (root.isEmpty()) {
			log.debug("no CUDA provisioning directory found -- using ONNX Runtime as shipped");
			return;
		}
		log.debug("CUDA provisioning: {}", root.get());
		List<Path> libraries = sonames(root.get());

		Optional<Path> ortDir = libraries.stream()
			.filter(p -> p.getFileName().toString().equals("libonnxruntime.so.1"))
			.map(Path::getParent)
			.findFirst();
		ortDir.ifPresent(CudaNatives::useNatives);

		// Everything except ONNX Runtime's own libraries, which it loads itself once pointed at
		// them. Loading those here as well would leave the same objects open under two handles.
		preload(libraries.stream()
			.filter(p -> ortDir.isEmpty() || !p.getParent().equals(ortDir.get()))
			.toList());
	}

	/** The provisioning directory, if there is one. */
	private static Optional<Path> locate() {
		String explicit = System.getProperty(DIR_PROPERTY, System.getenv(DIR_ENV));
		if (explicit != null && !explicit.isBlank()) {
			Path p = Path.of(explicit);
			if (!Files.isDirectory(p)) {
				log.warn("{} points at {}, which is not a directory -- ignoring",
					System.getProperty(DIR_PROPERTY) != null ? DIR_PROPERTY : DIR_ENV, p);
				return Optional.empty();
			}
			return Optional.of(p);
		}
		// From the working directory upward, so a test fork in yunet/ and a build from the reactor
		// root both find the same thing.
		Path dir = Path.of(System.getProperty("user.dir", ".")).toAbsolutePath();
		for (int i = 0; i < SEARCH_DEPTH && dir != null; i++, dir = dir.getParent()) {
			Path candidate = dir.resolve(LOCAL_DIR);
			if (Files.isDirectory(candidate)) {
				return Optional.of(candidate);
			}
		}
		// Installed once for the whole machine, which is what an application depending on this
		// library rather than building it wants.
		Path global = Path.of(System.getProperty("user.home", "."), ".facedetect4j", "cuda");
		return Files.isDirectory(global) ? Optional.of(global) : Optional.empty();
	}

	/**
	 * Every versioned shared library under the provisioning directory. Deliberately content-driven
	 * rather than reading a fixed layout: the directory names come from NVIDIA's tarballs
	 * ({@code cudnn-linux-x86_64-9.25.0.15_cuda13-archive}) and change with every release.
	 */
	private static List<Path> sonames(Path root) {
		try (Stream<Path> walk = Files.walk(root, 4)) {
			return walk.filter(Files::isRegularFile)
				.filter(p -> p.getFileName().toString().matches(SONAME))
				.sorted()
				.toList();
		} catch (IOException e) {
			log.warn("cannot scan {}: {}", root, e.toString());
			return List.of();
		}
	}

	private static void useNatives(Path dir) {
		String existing = System.getProperty(ORT_NATIVE_PATH);
		if (existing != null && !existing.isBlank()) {
			log.debug("{} already set to {} -- leaving it alone", ORT_NATIVE_PATH, existing);
			return;
		}
		if (!Files.isRegularFile(dir.resolve("libonnxruntime4j_jni.so"))) {
			// The JNI shim only ships in the Maven jar. Without it ONNX Runtime finds the directory,
			// loads the core, and then cannot bind a single method.
			log.warn("{} has no libonnxruntime4j_jni.so -- not using it. Re-run setup-cuda.sh.", dir);
			return;
		}
		System.setProperty(ORT_NATIVE_PATH, dir.toString());
		log.debug("ONNX Runtime natives: {}", dir);
	}

	/**
	 * Loads each library, retrying until a pass loads nothing new. Failures on the final pass are
	 * reported at debug only: the set deliberately includes libraries that may be unusable here
	 * (a CUDA 12 runtime fetched earlier, say, on a host since upgraded), and one that will not
	 * load is only a problem if something later asks for it — at which point the failure is
	 * reported with the context to act on.
	 */
	private static void preload(List<Path> libraries) {
		Set<Path> pending = new LinkedHashSet<>(libraries);
		List<String> failures = new ArrayList<>();
		while (!pending.isEmpty()) {
			failures.clear();
			int before = pending.size();
			for (var it = pending.iterator(); it.hasNext();) {
				Path lib = it.next();
				try {
					System.load(lib.toString());
					it.remove();
				} catch (UnsatisfiedLinkError | SecurityException e) {
					failures.add(lib.getFileName() + ": " + e.getMessage());
				}
			}
			if (pending.size() == before) {
				break;
			}
		}
		log.debug("preloaded {} of {} native libraries", libraries.size() - pending.size(),
			libraries.size());
		failures.forEach(f -> log.debug("not preloaded -- {}", f));
	}
}
