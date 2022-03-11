# environment.yml file is provided for conda

# How to launch

python main.py --help # print all the possible arguments

python main.py --matching_mode MATCHING_MODE -o OUTPUT_IMAGE_PATH --feathering img1 img2

--feathering # use feathering
--no-feathering # don't use feathering, just stitch

--matching_mode: baseline or superglue