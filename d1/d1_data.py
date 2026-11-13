"""Calibration and eval requests for d1-2B (d1_quant.py, d1_calib.py, the NPU validation): pictures from a COCO
parquet, Wikipedia passages, home-assistant commands, each with a set of noul/choice/score questions. The eval set
uses other pictures, passages and commands than the calibration set.

    CAL, EVAL = requests(m, COCO, WIKI, img_dir)      # [(state, {name: question}, features or None, image path or None)]
"""
import io, random
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

N_CAL_IMG, N_EVAL_IMG, N_CAL_TXT, N_EVAL_TXT = 64, 24, 48, 20

PIC = [
    {"type": "noul", "instructions": "Are there animals in the picture?"},
    {"type": "noul", "instructions": "Is there a vehicle (car, bus, bicycle, truck, train, boat) in the picture?"},
    {"type": "noul", "instructions": "Is anyone eating or drinking?"},
    {"type": "noul", "instructions": "Was the picture taken at night?"},
    {"type": "noul", "instructions": "Is there a person at a door or entrance?"},
    {"type": "noul", "instructions": "Is there any text or sign that can be read?"},
    {"type": "score", "instructions": "How many people are in the picture?", "criteria": ["0", "1", "2", "3", "4", "5 or more"]},
    {"type": "score", "instructions": "How cluttered is the scene?", "criteria": ["tidy", "some objects", "very cluttered"]},
    {"type": "choice", "instructions": "Where was the picture taken?",
     "criteria": {"indoors": "Inside a building", "street": "A street or road", "nature": "Nature, field, beach or sea", "other": "Somewhere else"}},
    {"type": "choice", "instructions": "Which room is it?",
     "criteria": {"kitchen": None, "living_room": None, "bedroom": None, "bathroom": None, "office": None, "not_a_room": "Not inside a home"}},
    {"type": "choice", "instructions": "What is the main subject?",
     "criteria": {"people": None, "animals": None, "food": None, "vehicles": None, "objects": None, "landscape": None}},
    {"type": "choice", "multiple": True, "instructions": "Which of these are visible?",
     "criteria": {"person": None, "dog_or_cat": "A dog or a cat", "car": None, "table": None, "plant": None}},
    {"type": "score", "instructions": "How bright is the picture?", "criteria": ["dark", "normal", "bright"]},
]
TXT = [
    {"type": "choice", "instructions": "What is the text mainly about?",
     "criteria": {"history": None, "science": None, "sports": None, "arts": "Music, film, literature or art", "geography": None, "politics": None, "other": None}},
    {"type": "noul", "instructions": "Does the text mention a specific year?"},
    {"type": "noul", "instructions": "Is the text about a person?"},
    {"type": "score", "instructions": "How technical is the text?", "criteria": ["not technical", "somewhat technical", "very technical"]},
    {"type": "choice", "multiple": True, "instructions": "Which of these does the text mention?",
     "criteria": {"person": "A named person", "place": "A named place", "organization": "A named organization", "number": "A quantity or measurement"}},
    {"type": "noul", "instructions": "Does the text describe a war or a battle?"},
]
CMD_Q = [
    {"type": "choice", "instructions": "What does the user want?",
     "criteria": {"turn_on": "Turn on or open something", "turn_off": "Turn off or close something", "set_value": "Set a temperature, brightness or volume",
                  "query": "Ask about a state or value", "media": "Play, pause or skip music or video", "timer": "A timer or reminder", "other": None}},
    {"type": "choice", "instructions": "Which area is it about?",
     "criteria": {"kitchen": None, "living_room": None, "bedroom": None, "bathroom": None, "garage": None, "garden": None, "none": "No area named"}},
    {"type": "noul", "instructions": "Is the user asking a question?"},
    {"type": "score", "instructions": "How urgent is it?", "criteria": ["not urgent", "soon", "immediately"]},
]
CMDS = [
    "Turn on the kitchen light", "Apaga el llum del menjador", "Enciende la luz del dormitorio", "Close the garage door",
    "What's the temperature in the living room?", "Posa la calefacció a 21 graus", "Pon música en el salón",
    "Set a timer for ten minutes", "Is the front door locked?", "Obre les persianes de l'habitació", "Baja el volumen",
    "Turn off everything in the bedroom", "Quina temperatura fa a fora?", "¿Está encendida la luz del baño?",
    "Remind me to water the garden at six", "Engega el reg del jardí", "Dim the living room lights to 30 percent",
    "Para la música", "There is smoke in the kitchen!", "Atura l'aspiradora", "Skip this song", "Open the garage",
    "¿Cuánta humedad hay en el baño?", "Tanca totes les llums", "Turn the heating off, I'm leaving", "Activa l'alarma",
    "Is the washing machine done?", "Apaga la tele del salón", "Encén el ventilador de l'habitació", "Lock the back door",
    "How much energy did we use today?", "Pon un temporizador de cinco minutos", "Water is leaking in the bathroom, help",
    "Switch on the porch light", "Abaixa la temperatura del dormitori", "Play the news", "Quin volum té l'altaveu?",
    "Turn on the coffee machine", "Cierra la puerta del garaje", "Set the bedroom to 19 degrees",
]


def sample_qs(pool, rng, k):
    return {f"q{j}": q for j, q in enumerate(rng.sample(pool, k))}



def requests(m, coco, wiki, img_dir):
    """m: a D1Ref (its picture() gives the fp32 features)."""
    img_dir = Path(img_dir); img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    rows = pq.read_table(coco, columns=["image"])["image"].to_pylist()
    cal_idx = random.Random(0).sample(range(len(rows)), N_CAL_IMG + 200)
    eval_idx = [i for i in random.Random(7).sample(range(len(rows)), 400) if i not in set(cal_idx)][:N_EVAL_IMG]
    cal_idx = cal_idx[:N_CAL_IMG]

    def pic(r):
        p = img_dir / f"{r}.jpg"
        if not p.exists():
            Image.open(io.BytesIO(rows[r]["bytes"])).convert("RGB").save(p, quality=92)
        return m.picture(str(p)), str(p)

    paras = [t.strip() for t in pq.read_table(wiki)["text"].to_pylist() if len(t.strip()) > 400 and not t.strip().startswith("=")]
    random.Random(0).shuffle(paras)
    cal_txt, eval_txt = paras[:N_CAL_TXT], paras[N_CAL_TXT:N_CAL_TXT + N_EVAL_TXT]
    cmds = CMDS[:]; random.Random(0).shuffle(cmds)
    cal_cmd, eval_cmd = cmds[:24], cmds[24:]
    CAL, EVAL = [], []
    for r in cal_idx:
        CAL.append((None, sample_qs(PIC, rng, rng.randint(4, 9)), *pic(r)))
    for p in cal_txt:
        CAL.append((p[:1500], sample_qs(TXT, rng, rng.randint(3, 6)), None, None))
    for c in cal_cmd:
        CAL.append((c, sample_qs(CMD_Q, rng, 4), None, None))
    for r in eval_idx:
        EVAL.append((None, {f"q{j}": q for j, q in enumerate(PIC)}, *pic(r)))
    for p in eval_txt:
        EVAL.append((p[:1500], {f"q{j}": q for j, q in enumerate(TXT)}, None, None))
    for c in eval_cmd:
        EVAL.append((c, {f"q{j}": q for j, q in enumerate(CMD_Q)}, None, None))
    return CAL, EVAL
