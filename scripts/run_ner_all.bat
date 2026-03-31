@echo off
cd /d "C:\Users\rjjos\Documents\NEU\Courses\Sem-2\NLP\ShipOfTheseus-NLP"
call venv\Scripts\activate.bat

echo Starting NER extraction at %date% %time%
python run_ner_single.py text_T0
echo Done text_T0 at %time%

python run_ner_single.py text_chatgpt
echo Done text_chatgpt at %time%

python run_ner_single.py text_dipper_high
echo Done text_dipper_high at %time%

python run_ner_single.py text_dipper_low
echo Done text_dipper_low at %time%

python run_ner_single.py text_pegasus_slight
echo Done text_pegasus_slight at %time%

python run_ner_single.py text_pegasus_full
echo Done text_pegasus_full at %time%

echo ALL DONE at %date% %time%
echo COMPLETE > data\processed\ner_extraction_done.flag
