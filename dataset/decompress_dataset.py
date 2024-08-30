import argparse
import tarfile
import requests
import os

def process_file(input_file: str) -> None:
    """
    Extract and save video files from a tar.gz archive.

    This function takes a file path or a URL as input, decompresses it,
    and saves the video files.

    :param input_file: The input file path or URL to process.
    """

    # Check if the input is a URL
    if input_file.startswith(('http://', 'https://')):
        # Stream the file from the URL
        response = requests.get(input_file, stream=True)
        tar_stream = tarfile.open(fileobj=response.raw, mode="r|gz")  # 'r|gz' for streaming mode
    else:
        # If the input is a file path
        tar_stream = tarfile.open(input_file, mode="r|gz")  # 'r|gz' for streaming mode

    # Process each file inside the tarball as it is streamed
    for member in tar_stream:
        if member.isfile():
            f = tar_stream.extractfile(member)
            if f is not None:
                # Construct a file name to save the video
                output_filename = os.path.join("output_videos", member.name)
                
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                
                # Write the video content to the file
                with open(output_filename, 'wb') as out_file:
                    out_file.write(f.read())
                
                print(f"Extracted and saved: {output_filename}")

    tar_stream.close()

def main():
    """
    Main function to parse command line arguments and process the input file.
    """
    parser = argparse.ArgumentParser(description='Extract video files from a tar.gz archive.')
    parser.add_argument('--input-file', '-i', type=str, required=True, help='The input file to process')

    args = parser.parse_args()

    process_file(args.input_file)

if __name__ == "__main__":
    main()
