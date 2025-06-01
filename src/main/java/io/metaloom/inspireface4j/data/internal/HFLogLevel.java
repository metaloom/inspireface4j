package io.metaloom.inspireface4j.data.internal;

public enum HFLogLevel {

	HF_LOG_NONE, // No logging, disables all log output
	HF_LOG_DEBUG, // Debug level for detailed system information mostly useful for developers
	HF_LOG_INFO, // Information level for general system information about operational status
	HF_LOG_WARN, // Warning level for non-critical issues that might need attention
	HF_LOG_ERROR, // Error level for error events that might still allow the application to continue running
	HF_LOG_FATAL; // Fatal level for severe error events that will presumably lead the application to abort

}
