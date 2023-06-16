#include "DynamicLibrary.h"
#include <vector>
#include <cassert>
#if defined(CUDAQ_HAVE_DLFCN_H)
#include <dlfcn.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

#if defined(_WIN32)
namespace {
struct Globals {
  void *OpenedHandles;
};

Globals &getGlobals() {
  static Globals G;
  return G;
}

void *IsOpenedHandlesInstance(void *Handle) {
  auto &Inst = getGlobals().OpenedHandles;
  return Handle == &Inst ? &Inst : nullptr;
}

bool GetProcessModules(HANDLE H, DWORD &Bytes, HMODULE *Data = nullptr) {
  if (!EnumProcessModules(H, Data, Bytes, &Bytes)) {
    return false;
  }
  return true;
}

// Returns the last Win32 error, in string format. Returns an empty string if
// there is no error.
std::string GetLastErrorAsString() {
  // Get the error message ID, if any.
  DWORD errorMessageID = ::GetLastError();
  if (errorMessageID == 0) {
    return std::string(); // No error message has been recorded
  }

  LPSTR messageBuffer = nullptr;

  // Ask Win32 to give us the string version of that message ID.
  // The parameters we pass in, tell Win32 to create the buffer that holds the
  // message for us (because we don't yet know how long the message string will
  // be).
  size_t size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR)&messageBuffer, 0, NULL);

  // Copy the error message into a std::string.
  std::string message(messageBuffer, size);

  // Free the Win32's string's buffer.
  LocalFree(messageBuffer);

  return message;
}
} // namespace
#endif

void *cudaq::DynamicLibrary::DLOpen(const char *Filename, std::string *Err) {
#if defined(CUDAQ_HAVE_DLFCN_H)
  void *Handle = ::dlopen(Filename, RTLD_LAZY | RTLD_GLOBAL);
  if (!Handle) {
    if (Err)
      *Err = ::dlerror();
    return nullptr
  } else {
    return Handle;
  }
#elif defined(_WIN32)
  if (Filename == nullptr) {
      return &getGlobals().OpenedHandles;
  } else {
      std::wstring filename_ws = std::wstring(Filename, Filename + strlen(Filename));
    LPCWSTR filename_wc_ptr = filename_ws.c_str();
    auto hModule = LoadLibraryW(filename_wc_ptr);
    if (hModule == nullptr) {
      if (Err)
        *Err = GetLastErrorAsString();
      return nullptr;
    } else {
      return reinterpret_cast<void *>(hModule);
    }
  }
#endif
  return nullptr;
}

void cudaq::DynamicLibrary::DLClose(void *Handle) {
#if defined(CUDAQ_HAVE_DLFCN_H)
  ::dlclose(Handle);
#elif defined(_WIN32)
  if (!IsOpenedHandlesInstance(Handle))
    FreeLibrary((HMODULE)Handle);
#endif
}

void *cudaq::DynamicLibrary::DLSym(void *Handle, const char *Symbol) {
#if defined(CUDAQ_HAVE_DLFCN_H)
  return ::dlsym(Handle, Symbol);
#elif defined(_WIN32)
  auto *handle_cast = IsOpenedHandlesInstance(Handle);
  if (!handle_cast) {
    return (void *)uintptr_t(GetProcAddress((HMODULE)Handle, Symbol));
  }

  DWORD Bytes = 0;
  HMODULE Self = HMODULE(GetCurrentProcess());
  if (!GetProcessModules(Self, Bytes))
    return nullptr;

  // Get the most recent list in case any modules added/removed between calls
  // to EnumProcessModulesEx that gets the amount of, then copies the HMODULES.
  // MSDN is pretty clear that if the module list changes during the call to
  // EnumProcessModulesEx the results should not be used.
  std::vector<HMODULE> Handles;
  do {
    assert(Bytes && ((Bytes % sizeof(HMODULE)) == 0) &&
           "Should have at least one module and be aligned");
    Handles.resize(Bytes / sizeof(HMODULE));
    if (!GetProcessModules(Self, Bytes, Handles.data()))
      return nullptr;
  } while (Bytes != (Handles.size() * sizeof(HMODULE)));

  // Try EXE first, mirroring what dlsym(dlopen(NULL)) does.
  if (FARPROC Ptr = GetProcAddress(HMODULE(Handles.front()), Symbol))
    return (void *)uintptr_t(Ptr);

  if (Handles.size() > 1) {
    // This is different behaviour than what Posix dlsym(dlopen(NULL)) does.
    // Doing that here is causing real problems for the JIT where msvc.dll
    // and ucrt.dll can define the same symbols. The runtime linker will choose
    // symbols from ucrt.dll first, but iterating NOT in reverse here would
    // mean that the msvc.dll versions would be returned.
    for (auto I = Handles.rbegin(), E = Handles.rend() - 1; I != E; ++I) {
      // Get filename with full path for current process EXE
      /*wchar_t filename[MAX_PATH];
      DWORD result =
          ::GetModuleFileName(HMODULE(*I), filename, _countof(filename));
      std::wcout << "Module name: " << std::wstring(filename) << "\n";*/

      if (FARPROC Ptr = GetProcAddress(HMODULE(*I), Symbol))
        return (void *)uintptr_t(Ptr);
    }
  }
  return nullptr;
#endif
  return nullptr;
}

void cudaq::DynamicLibrary::ForEachLinkedLib(
    std::function<bool(const std::string &)> callback) {
  DWORD Bytes = 0;
  HMODULE Self = HMODULE(GetCurrentProcess());
  if (!GetProcessModules(Self, Bytes))
    return;

  // Get the most recent list in case any modules added/removed between calls
  // to EnumProcessModulesEx that gets the amount of, then copies the HMODULES.
  // MSDN is pretty clear that if the module list changes during the call to
  // EnumProcessModulesEx the results should not be used.
  std::vector<HMODULE> Handles;
  do {
    assert(Bytes && ((Bytes % sizeof(HMODULE)) == 0) &&
           "Should have at least one module and be aligned");
    Handles.resize(Bytes / sizeof(HMODULE));
    if (!GetProcessModules(Self, Bytes, Handles.data()))
      return;
  } while (Bytes != (Handles.size() * sizeof(HMODULE)));

  for (auto I = Handles.begin(), E = Handles.end(); I != E; ++I) {
    // Get filename with full path
    wchar_t filename[MAX_PATH];
    DWORD result =
        ::GetModuleFileName(HMODULE(*I), filename, _countof(filename));
    const auto wFullLibFilename = std::wstring(filename);
    const std::string fullLibFilename(wFullLibFilename.begin(),
                                      wFullLibFilename.end());
    const bool cb_result = callback(fullLibFilename);
    if (!cb_result) {
      break;
    }
  }
}