import torchvision
import torchvision.models as models

def main():
    model = torchvision.models.mobilenet_v3_large(pretrained=True)


if __name__ == "__main__":
    main()