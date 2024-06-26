SCORING_MAPPING = {
    1: "Strongly disagree",
    2: "Somewhat disagree",
    3: "Neutral/agnostic",
    4: "Somewhat agree",
    5: "Strongly agree",
}

PREDICTIONS_MAPPING = {
    1: "Very unpromising",
    2: "Somewhat unpromising",
    3: "Unsure/agnostic",
    4: "Somewhat promising",
    5: "Very promising",
}

BIG_FIVE_COLUMNS = [
    "e1",
    "c1",
    "n1",
    "a1",
    "o1",
    "optimism1",
    "o2",
    "n2",
    "a2",
    "a3",
    "c2",
    "optimism2",
    "c3",
    "e2",
    "c4",
    "e3",
    "c5",
    "optimism3",
    "o3",
    "n3",
    "n4",
    "o4",
    "e4",
    "optimism4",
]

BIG_FIVE_CATEGORIES = {
    "e": "Extraversion",
    "c": "Conscientiousness",
    "n": "Neuroticism",
    "a": "Agreeableness",
    "o": "Openness",
    "optimism": "Optimism",
}

MORAL_FOUNDATIONS_CATEGORIES = {
    "risk": "Risk Taking",
    "selfcontrol": "Self-Control",
    "trad": "Traditionalism",
    "communal": "Communalism",
    "compassion": "Compassion",
    "liberty": "Liberty",
}

MORAL_FOUNDATIONS_COLUMNS = [
    "risk1",
    "selfcontrol1",
    "risk3",
    "trad1",
    "risk4",
    "communal2",
    "compassion2",
    "trad4",
    "trad5",
    "trad2",
    "risk2",
    "trad3",
    "risk5",
    "compassion5",
    "liberty3",
    "compassion3",
    "risk6",
    "communal3",
    "selfcontrol2",
    "compassion4",
    "compassion1",
    "risk7",
    "liberty2",
    "liberty4",
    "liberty1",
    "communal1",
    "liberty5",
    "risk8",
    "selfcontrol3",
    "communal5",
    "selfcontrol4",
    "selfcontrol5",
    "communal4",
    "selfcontrol6",
]


ACCEPTED_RESPONSES = {
    "How would you describe your political orientation?": [
        "Centrist",
        "Libertarian",
        "Moderately conservative",
        "Moderately progressive",
        "Strongly progressive",
        "Strongly conservative",
    ]
}

MULTIPLE_ACCEPTED_RESPONSES = {
    "Please select which of the following EA cause areas you are actively involved in.*": [
        "Not actively involved* in any of the below",
        "Global health and development",
        "Farmed/wild animal welfare",
        "Existential risk (general)",
        "Biosecurity & pandemics",
        "Philosophical work",
        "AI risk",
        "Fieldbuilding/community development",
        "EA-related policy work",
        "Cause prioritization and/or effective giving",
        "I spent at least 5h/week on EA, but not related to any one specific cause area",
    ],
}


EA_QUESTION_PAIRS = {
    # EA
    "Please broadly indicate how much promise you see in the pursuit of each of the following EA cause areas. [Global health and development]": "Please broadly indicate how much promise you think the EA community as a whole sees in the pursuit of each of the following EA cause areas. [Global health and development]",
    "Please broadly indicate how much promise you see in the pursuit of each of the following EA cause areas. [Farmed/wild animal welfare]": "Please broadly indicate how much promise you think the EA community as a whole sees in the pursuit of each of the following EA cause areas. [Farmed/wild animal welfare]",
    "Please broadly indicate how much promise you see in the pursuit of each of the following EA cause areas. [Existential risk (general)]": "Please broadly indicate how much promise you think the EA community as a whole sees in the pursuit of each of the following EA cause areas. [Existential risk (general)]",
    "Please broadly indicate how much promise you see in the pursuit of each of the following EA cause areas. [Biosecurity & pandemics]": "Please broadly indicate how much promise you think the EA community as a whole sees in the pursuit of each of the following EA cause areas. [Biosecurity & pandemics]",
    "Please broadly indicate how much promise you see in the pursuit of each of the following EA cause areas. [Philosophical work]": "Please broadly indicate how much promise you think the EA community as a whole sees in the pursuit of each of the following EA cause areas. [Philosophical work]",
    "Please broadly indicate how much promise you see in the pursuit of each of the following EA cause areas. [AI risk]": "Please broadly indicate how much promise you think the EA community as a whole sees in the pursuit of each of the following EA cause areas. [AI risk]",
    "Please broadly indicate how much promise you see in the pursuit of each of the following EA cause areas. [Fieldbuilding/community development]": "Please broadly indicate how much promise you think the EA community as a whole sees in the pursuit of each of the following EA cause areas. [Fieldbuilding/community development]",
    "Please broadly indicate how much promise you see in the pursuit of each of the following EA cause areas. [EA-related policy work]": "Please broadly indicate how much promise you think the EA community as a whole sees in the pursuit of each of the following EA cause areas. [EA-related policy work]",
    "Please broadly indicate how much promise you see in the pursuit of each of the following EA cause areas. [Cause prioritization/effective giving]": "Please broadly indicate how much promise you think the EA community as a whole sees in the pursuit of each of the following EA cause areas. [Cause prioritization/effective giving]",
    # Promise
    "I believe effective altruism is a force for good in the world": "I think the EA community as a whole believes that effective altruism is a force for good in the world",
    "I believe effective altruism has a good reputation": "I think the EA community as a whole believes that effective altruism has a good reputation",
    "I believe the community aspect is essential to effective altruism": "I think the EA community as a whole believes that the community aspect is essential to effective altruism",
    "I proudly identify as an effective altruist": "I think the EA community as a whole proudly identifies as effective altruists",
    "I think longtermist causes should be the primary focus in effective altruism": "I think the EA community as a whole believes that longtermist causes should be the primary focus in effective altruism",
    "I have a positive view of effective altruism's overall shift towards longtermist causes": "I think the EA community as a whole has a positive view of effective altruism's overall shift towards longtermist causes",
    "I think the FTX crisis was a reflection of deeper problems with the philosophy and/or community of effective altruism": "I think the EA community as a whole believes that the FTX crisis was a reflection of deeper problems with the philosophy and/or community of effective altruism",
    # Forced choice
    "If I were forced to choose, I would say that the three qualities that render a person most effective/impactful (of the following list) are": "If the EA community as a whole were forced to choose, I believe they would say would say that the three qualities that render a person most effective/impactful (of the following list) are",
    "If I were forced to choose, I would say that the three areas where upskilling would lead to the highest impact (of the following list) are": "If the EA community as a whole were forced to choose, I believe they would say that the three areas where upskilling would lead to the highest impact (of the following list) are",
    "If I were forced to choose, I would say that the single quality that renders a person most effective/impactful (of the following list) is": "If the EA community as a whole were forced to choose, I believe they would say that the single quality that renders a person most effective/impactful (of the following list) is",
    "If I were forced to choose, I would say that the single area where upskilling would lead to the highest impact (of the following list) is": "If the EA community as a whole were forced to choose, I believe they would say that the single area where upskilling would lead to the highest impact (of the following list) is",
}

ALIGNMENT_QUESTION_PAIRS = {
    "Please broadly indicate how much promise you see in each of the following buckets of alignment approaches, as described in Shallow review of live agendas in alignment & safety.  [Understand existing models (evals, interp, science of deep learning)]": "Please indicate how much promise you think the alignment research community as a whole sees in each of the following buckets of alignment approaches, as described in Shallow review of live agendas in alignment & safety.  [Understand existing models (evals, interp, science of deep learning)]",
    "Please broadly indicate how much promise you see in each of the following buckets of alignment approaches, as described in Shallow review of live agendas in alignment & safety.  [Control the AI (deception, model edits, value alignment, goal robustness)]": "Please indicate how much promise you think the alignment research community as a whole sees in each of the following buckets of alignment approaches, as described in Shallow review of live agendas in alignment & safety.  [Control the AI (deception, model edits, value alignment, goal robustness)]",
    "Please broadly indicate how much promise you see in each of the following buckets of alignment approaches, as described in Shallow review of live agendas in alignment & safety.  [Make AI solve it (scalable oversight, cyborgism, self-alignment)]": "Please indicate how much promise you think the alignment research community as a whole sees in each of the following buckets of alignment approaches, as described in Shallow review of live agendas in alignment & safety.  [Make AI solve it (scalable oversight, cyborgism/human-AI enhancements, self-alignment.)]",
    "Please broadly indicate how much promise you see in each of the following buckets of alignment approaches, as described in Shallow review of live agendas in alignment & safety.  [Theory work (agency, corrigibility, ontology, framework building)]": "Please indicate how much promise you think the alignment research community as a whole sees in each of the following buckets of alignment approaches, as described in Shallow review of live agendas in alignment & safety.  [Theory work (agency, corrigibility, ontology, framework building)]",
    "I currently support pausing or dramatically slowing AI development": "I believe that the alignment research community supports pausing or dramatically slowing AI development",
    "Advancing AI capabilities and doing alignment research are mutually exclusive goals": "I believe that the alignment research community thinks that advancing AI capabilities and doing alignment research are mutually exclusive goals",
    "The alignment problem is mainly about control rather than coexistence": "I believe that the alignment research community thinks that the alignment problem is mainly about control rather than coexistence",
    "Alignment research that has some probability of also advancing capabilities should not be done": "I believe that the alignment research community thinks that alignment research that has some probability of also advancing capabilities should not be done",
    "I expect there will be AGI in the next five years": "I believe that the alignment research community expects there will be AGI in the next five years",
    "I expect there will be superintelligent AI in the next five years": "I believe that the alignment research community expects there will be superintelligent AI in the next five years",
    "We will have made significant progress in alignment before we get AGI": "I believe that the alignment research community thinks we will have made significant progress in alignment before we get AGI",
    "Current alignment research is on track to solve alignment before we get AGI": "I believe that the alignment research community thinks that current alignment research is on track to solve alignment before we get AGI",
    "The current alignment research landscape covers the space of plausible approaches to alignment well": "I believe that the alignment research community thinks that the current alignment research landscape covers the space of plausible approaches to alignment well",
    "Alignment should be a more multidisciplinary field": "I believe that the alignment research community thinks that alignment should become a more multidisciplinary field",
    "Alignment researchers require a computer science, mathematics, physics, engineering, or related  background": "I believe that the alignment research community thinks that alignment researchers require a computer science, mathematics, physics, engineering, or related  background",
    "Field-level success in alignment requires ensuring that the right kinds of thinkers are involved in the field": "I believe that the alignment research community thinks that field-level success in alignment requires ensuring that the right kinds of thinkers are involved in the field",
    "If the government were to provide significant funding for advancing alignment research, I would be...": "If the government were to provide significant funding for advancing alignment research, I believe that the alignment research community would be...",
    "If I were forced to choose, I would say that the three most important properties that a good alignment researcher should have (of the following list) are": "If the alignment research community as a whole were forced to choose, I believe they would say would say that the three most important properties that a good alignment researcher should have (of the following list) are",
    "If I were forced to choose, I would say that the single most important property that a good alignment researcher should have (of the following list) is": "If the alignment research community as a whole were forced to choose, I believe they would say that the single most important property that a good alignment researcher should have (of the following list) is",
}

QUESTION_PAIRS = EA_QUESTION_PAIRS | ALIGNMENT_QUESTION_PAIRS

FLIPPED_DELAY_DISCOUNTING_SCORES = {
    "$30 tonight": "0",
    "$85 in 14 days": "1",
    "$40 tonight": "0",
    "$55 in 25 days": "1",
    "$67 tonight": "0",
    "$85 in 35 days": "1",
    "$15 tonight": "0",
    "$35 in 10 days": "1",
    "$83 tonight": "0",
    "$85 in 35 days": "1",
    "$21 tonight": "0",
    "$30 in 75 days": "1",
    "$50 tonight": "0",
    "$80 in 70 days": "1",
    "$48 tonight": "0",
    "$55 in 45 days": "1",
    "$25 tonight": "0",
    "$35 in 25 days": "1",
    "$53 tonight": "0",
    "$55 in 55 days": "1",
}

K_VALUES = {
    "delaydiscounting1": {"0": 0, "1": 0.13095238095238096},
    "delaydiscounting2": {"0": 0, "1": 0.015},
    "delaydiscounting3": {"0": 0, "1": 0.007675906183368873},
    "delaydiscounting4": {"0": 0, "1": 0.13333333333333336},
    "delaydiscounting5": {"0": 0, "1": 0.0006884681583476786},
    "delaydiscounting6": {"0": 0, "1": 0.002666666666666666},
    "delaydiscounting7": {"0": 0, "1": 0.008571428571428572},
    "delaydiscounting8": {"0": 0, "1": 0.003240740740740739},
    "delaydiscounting9": {"0": 0, "1": 0.015999999999999997},
    "delaydiscounting10": {"0": 0, "1": 0.0006861063464837048},
}

PREDICTIONS = {
    # EA
    "ep1": "Global health and development",
    "ep2": "Farmed/wild animal welfare",
    "ep3": "Existential risk (general)",
    "ep4": "Biosecurity & pandemics",
    "ep5": "Philosophical work",
    "ep6": "AI risk",
    "ep7": "Fieldbuilding/community development",
    "ep8": "EA-related policy work",
    "ep9": "Cause prioritization/effective giving",
    # Alignment
    "ap1": "Understand existing models",
    "ap2": "Control the AI",
    "ap3": "Make AI solve it",
    "ap4": "Theory work",
}
