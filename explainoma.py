from flask import Flask, request, render_template
from utils.inference import load_model, predict
from skimage import measure
from models import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

model = load_model("static/models/model.pth")

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        try:
            extension = file.filename.split(".")[-1].lower()
        except Exception as e:
            return

        if extension in ['jpg', 'jpeg']:

            image, output, attributions, prediction, explanation_abbr, explanation, detailed_explanation, confidence = predict(file, model)

            diagnosis = "Die Läsion wurde als "+prediction+" vorhergesagt."

            """ A """
            #attr = attributions[topk_indices[0]]
            #attr_non_zero = attr[attr > 0.]
            #if len(attr_non_zero) == 0:
            #    perc = 0
            #else:
            #    perc = (np.percentile(attr_non_zero, 85))
            #attr = (attr >= perc)

            fig, ax = plt.subplots()
            ax.imshow(image)
            #contours = measure.find_contours(attr)
            #for contour in contours:
            #    ax.plot(contour[:, 1], contour[:, 0], linewidth=0.9, color='white')
            plt.axis('off')
            #ax.set_title(labels_full[topk_indices[0]], fontsize=10, fontweight='bold', loc='center', wrap=True)
            plt.savefig('static/images/a.jpg', bbox_inches='tight')

            return render_template('index.html', prediction=prediction, attributions=attributions,
                                   explanation_abbr=explanation_abbr, explanation=explanation,
                                   detailed_explanation=detailed_explanation,
                                   confidence=confidence, diagnosis=diagnosis,
                                   url='/static/images/')
        else:
            return render_template('index.html')

    return render_template('index.html')


