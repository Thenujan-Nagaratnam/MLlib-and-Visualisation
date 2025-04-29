import os
import webbrowser

from h2o_wave import app, data, main, Q, ui
from pyspark.ml.tuning import CrossValidatorModel
from src.lyrics.services.pipelines.lr_pipeline import LogisticRegressionPipeline

DATASET_PATH = os.path.abspath("data/merged-dataset.csv")

MODEL_DIR_PATH = os.path.abspath("model/")

pipeline: LogisticRegressionPipeline
model: CrossValidatorModel


def on_startup():
    global pipeline
    global model

    if not (
        os.path.exists(MODEL_DIR_PATH)
        and os.path.isdir(MODEL_DIR_PATH)
        and len(os.listdir(MODEL_DIR_PATH)) > 0
    ):
        os.makedirs(MODEL_DIR_PATH, exist_ok=True)

        pipeline = LogisticRegressionPipeline()

        print("PLEASE WAIT UNTIL YOU SEE => INFO: Application startup complete.")

        model = pipeline.train_and_test(
            dataset_path=DATASET_PATH,
            train_ratio=0.8,
            store_model_on=MODEL_DIR_PATH,
            print_statistics=True,
        )
    else:
        pipeline = LogisticRegressionPipeline()

        print("PLEASE WAIT UNTIL YOU SEE => INFO: Application startup complete.")

        model = CrossValidatorModel.load(MODEL_DIR_PATH)

    webbrowser.open("http://localhost:10101/")


def on_shutdown():
    pipeline.stop()


@app("/", on_startup=on_startup, on_shutdown=on_shutdown)
async def serve(q: Q) -> None:
    if not q.client.init_app:
        q.page["meta"] = ui.meta_card(
            "",
            title="Lyrics Classifier",
            layouts=[
                ui.layout(
                    breakpoint="xs",
                    zones=[
                        ui.zone("header", size="100px"),
                        ui.zone(
                            name="body",
                            direction=ui.ZoneDirection.ROW,
                            justify=ui.ZoneJustify.CENTER,
                            wrap=ui.ZoneWrap.STRETCH,
                        ),
                    ],
                )
            ],
        )

        q.page["header"] = ui.header_card(
            box=ui.box(
                zone="header",
                width="100%",
                height="80px",
            ),
            title="Lyrics Classifier",
            subtitle="Empower your classification decisions with our app, \
            where we confidently categorize data into 8 classes, relying on \
            probabilities exceeding 35%, or identifying it as an \
            'UNKNOWN' type when certainty eludes us.",
            icon="MusicInCollection",
        )

        q.client.init_app = True

    q.page["input_form"] = ui.form_card(
        box=ui.box(
            zone="body",
            width="95%",
        ),
        items=[
            ui.textbox(
                name="input_lyrics_text_box",
                value=q.args.input_lyrics_text_box,
                placeholder="Enter a lyrics",
                multiline=True,
                height="300px",
            ),
            ui.button(
                name="predict_button",
                label="Predict",
                width="100%",
                primary=True,
            )
        ]
    )

    if (
        q.args.predict_button
        and q.args.input_lyrics_text_box
    ):
        q.page["meta"].dialog = ui.dialog(
            name="prediction_time_loading_dialog",
            title="",
            width="25%",
            items=[
                ui.progress(
                    label="Running prediction algorithm",
                ),
            ],
            blocking=True,
        )

        await q.page.save()

        threshold = 0.35
        prediction, probabilities = pipeline.predict_one(
            unknown_lyrics=q.args.input_lyrics_text_box,
            threshold=threshold,
            model=model,
        )

        q.page["meta"].dialog = None

        await q.page.save()

        if prediction == "UNKNOWN":
            result_dialog_content = f"""
                <div style="text-align: center; background-color: #ffffff; padding: 20px; border: 1px solid #000;">
                    <p style="font-size: 36px; color: #ff0000; font-weight: bold;">
                        Genre Prediction
                    </p>
                    <p style="font-size: 24px; color: #000;">
                        It's an {prediction.upper()} music since highest prediction 
                        probability less than the threshold probability 
                        ({max(probabilities.values()):.6f} < {threshold:.2f})
                    </p>
                </div>
            """
        else:
            result_dialog_content = f"""
                <div style="text-align: center; background-color: #ffffff; padding: 20px; border: 1px solid #000;">
                    <p style="font-size: 36px; color: #ff0000; font-weight: bold;">
                        Genre Prediction
                    </p>
                    <p style="font-size: 24px; color: #000;">
                        It's a {prediction.capitalize()} music with 
                        a probability of {probabilities[prediction] * 100:.6f}%
                    </p>
                </div>
            """

        q.page["meta"].dialog = ui.dialog(
            name="prediction_result_dialog",
            title="",
            width="50%",
            items=[
                ui.markup(result_dialog_content),
                ui.template(" "),
                ui.buttons(
                    items=[
                        ui.button(
                            name="close_result_button",
                            label="Close",
                            primary=True,
                        )
                    ],
                    justify=ui.ButtonsJustify.END,
                )
            ],
            blocking=False,
            closable=True,
        )

        pie_chart_dict = {
            max(probabilities, key=probabilities.get): probabilities[max(probabilities, key=probabilities.get)],
            "other": sum(v for k, v in probabilities.items() if k != max(probabilities, key=probabilities.get))
        }
        colors = ["#000000", "#36BFFA"]
        q.page["pie_chart_result"] = ui.wide_pie_stat_card(
            box=ui.box(
                zone="body",
                width="35%",
                height="500px"
            ),
            title="Analyzing the probability of the highest probability genre \
            against the sum of other genres' probabilities",
            pies=[
                ui.pie(
                    label=genre,
                    value=f"{probability * 100:.1f}%",
                    fraction=probability,
                    color=colors.pop(),
                )
                for genre, probability in pie_chart_dict.items()
            ]
        )

        q.page["bar_chart_result"] = ui.plot_card(
            box=ui.box(
                zone="body",
                width="59%",
            ),
            title="Genre probability distribution of the lyrics",
            data=data(
                'genre probability',
                len(probabilities),
                rows=[
                    (genre.capitalize(), probability * 100)
                    for genre, probability in probabilities.items()
                ],
            ),
            plot=ui.plot(
                marks=[
                    ui.mark(
                        type='interval',
                        x='=genre',
                        y='=probability',
                        y_min=0,
                        fill_color="#36BFFA",
                        stroke_color="#5E5E5E",
                    )
                ]
            ),
        )

    if q.args.close_result_button:
        q.page["meta"].dialog = None

    await q.page.save()
