{
  description = "CUDA development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";

    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      config.cudaSupport = true;
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        gcc
        gnumake
        cmake
        pkg-config

        libpng
        zlib
        tbb
        gbenchmark

        gst_all_1.gstreamer
        gst_all_1.gst-plugins-base
        gst_all_1.gst-plugins-good
        gst_all_1.gst-plugins-bad
        gst_all_1.gst-plugins-ugly
        gst_all_1.gst-libav
        ninja

        cudaPackages.cuda_cudart
        cudaPackages.cuda_nvcc
        cudaPackages.cudnn
        cudaPackages.nsight_compute
        cudaPackages.nsight_systems
      ];
      shellHook = ''
        export CUDA_PATH=${pkgs.cudaPackages.cuda_cudart}
        export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.cudaPackages.cuda_cudart}/lib:$LD_LIBRARY_PATH
        export PATH=${pkgs.cudaPackages.cuda_nvcc}/bin:$PATH

        export GST_PLUGIN_PATH=$(pwd)
        cmake -S . -B build --preset release -D USE_CUDA=ON
      '';
    };
  };
}
