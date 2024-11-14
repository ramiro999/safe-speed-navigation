import os

def find_images(directory):
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    images = [f for f in os.listdir(directory) if f.endswith(supported_extensions)]
    images.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return images

def main():
    directory = 'resultados_kitti'
    if os.path.exists(directory):
        images = find_images(directory)
        for image in images:
            print(image)
    else:
        print(f"Directory '{directory}' does not exist.")

if __name__ == "__main__":
    main()