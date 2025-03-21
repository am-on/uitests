{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs; [
    python312
    python312Packages.pytesseract
  ] ++ lib.optionals (pkgs.stdenv.isLinux) [ xvfb-run ]; # Install xvfb-run ONLY on Linux

  # Set up environment variables
  env = {
    DISPLAY = ":99"; # Required for GUI testing
  };

  # Run Xvfb automatically when entering the shell on Linux
  scripts.start_xvfb.exec = lib.optionalString pkgs.stdenv.isLinux "Xvfb :99 -screen 0 1920x1080x24 &";

  scripts.run.exec = "poetry run python src/ui-tests/main.py";

  # https://devenv.sh/languages/
  # languages.python.enable = true;
  languages.python.poetry = {
    enable = true;
    activate.enable = true;
    install.enable = true;
    install.quiet = true;
  };

  git-hooks.hooks = {
    # Python
    ruff.enable = true;
    ruff-format.enable = true;
    mypy.enable = true;
    # Nix
    nixpkgs-fmt.enable = true;
    # General
    trim-trailing-whitespace.enable = true;
    end-of-file-fixer.enable = true;
    typos.enable = true;
  };

}
