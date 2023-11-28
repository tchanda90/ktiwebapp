import torch
import torchvision.transforms as transforms
from skimage import transform
from PIL import Image
from models import *


def load_model(path):
    model = CharacteristicsClassifier()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict(file, model):
    image = np.array(Image.open(file))
    orig_img = transform.resize(image, (image_size, image_size))
    image = model.test_transform(image=image, mask=image)['image'].unsqueeze(0)
    output = model.base_model(image)

    attributions = torch.cat([model.attribute(output, class_idx) for class_idx in range(model.num_classes)], dim=1)
    attributions = F.interpolate(attributions, size=(image_size, image_size), mode='bilinear')
    attributions = attributions.detach().squeeze(0).numpy()

    output = output.squeeze(0)
    prediction = (output >= thresholds).numpy()

    print(torch.sigmoid(output))

    if np.sum(prediction[0:7]) == 7:
        prediction = 'Melanom'
        explanation_abbr = mel_class_labels[torch.argmax(output[0:7])]
        confidence = torch.max(output[0:7])
    else:
        prediction = 'Nävus'
        explanation_abbr = nev_class_labels[torch.argmax(output[7:])]
        confidence = torch.max(output[7:])

    detailed_explanation = descriptions[explanation_abbr]
    explanation = labels_mapping[explanation_abbr]

    """
    topk_values, topk_indices = torch.topk(output[7:], 2)
    topk_values = torch.sigmoid(topk_values)
    explanation = list(pd.Series(char_class_labels)[list(topk_indices)].values)
    confidence = round(topk_values.max().item(), 2)
    """

    return orig_img, output, attributions, prediction, explanation_abbr, explanation, detailed_explanation, confidence

