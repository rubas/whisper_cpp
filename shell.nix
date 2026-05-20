{
  pkgs ? import <nixpkgs> { },
}:
let
  rocmPath = pkgs.symlinkJoin {
    name = "rocm-path-whisper-cpp";
    paths = with pkgs.rocmPackages; [
      clr
      hipblas
      hipblas-common
      rocblas
      hip-common
      hipcc
      rocm-cmake
      rocm-device-libs
      rocm-runtime
      rocm-comgr
      rocm-core
      rocminfo
      llvm.clang
    ];
  };
in
pkgs.mkShell {
  packages = [
    rocmPath
    pkgs.cmake
    pkgs.pkg-config
    pkgs.perl
    pkgs.libclang
  ];

  shellHook = ''
    export ROCM_PATH=${rocmPath}
    export HIP_PATH=${rocmPath}
    export HIP_ROOT_DIR=${rocmPath}
    export ROCM_HOME=${rocmPath}
    export DEVICE_LIB_PATH=${rocmPath}/amdgcn/bitcode
    export HIP_DEVICE_LIB_PATH=${rocmPath}/amdgcn/bitcode
    export CMAKE_PREFIX_PATH=${rocmPath}:''${CMAKE_PREFIX_PATH:-}
    export LIBCLANG_PATH=${pkgs.libclang.lib}/lib
    export AMDGPU_TARGETS="gfx1100;gfx1101;gfx1102;gfx1103;gfx1200;gfx1201"
    export GPU_TARGETS="$AMDGPU_TARGETS"
    export PATH=${rocmPath}/bin:$PATH
    export LD_LIBRARY_PATH=${rocmPath}/lib:''${LD_LIBRARY_PATH:-}
    echo "rocm-path: $ROCM_PATH"
    echo "AMDGPU_TARGETS: $AMDGPU_TARGETS"
  '';
}
