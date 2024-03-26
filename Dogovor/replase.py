from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx import Document
from babel import numbers
import docx
from sympy import re
import raz


def find_paragraph(doc, target_text):
    for i, paragraph in enumerate(doc.paragraphs):
        if target_text in paragraph.text:
            p = doc.paragraphs[i]
            p.clear()  # Удаление текста из параграфа
            return i
    return -1


def replace_text(path, _nomDog_, _nameOrg_, _cal_, _predmetDog_, _fin_, _fioDir_, _osnovanie_, _adresOrg_, _doljnost_, _nameOrgSokr_, indicate):
    # -----Вносим тип договора----------------
    _text_ = raz.get_text(indicate)
    doc = docx.Document(path)
    index = find_paragraph(doc, "[_text_]")
    for text in reversed(_text_):
        text_paragraph = doc.paragraphs[index]
        new_paragraph = text_paragraph.insert_paragraph_before(text)
        new_paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    doc.save(path)  # Сохраняем документ после вставки новых параграфов
    replace_in_tables(path, _nomDog_, _nameOrg_,
                      _fioDir_, _adresOrg_, _doljnost_, _nameOrgSokr_)


# -----Вносим правки в договор-----------
def replace_in_tables(path, _nomDog_, _nameOrg_, _fioDir_, _adresOrg_, _doljnost_, _nameOrgSokr_):
    doc = docx.Document(path)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    if "[_nomDog_]" in paragraph.text:
                        paragraph.text = paragraph.text.replace(
                            "[_nomDog_]", _nomDog_)
                    if "[_nameOrg_]" in paragraph.text:
                        paragraph.text = paragraph.text.replace(
                            "[_nameOrg_]", _nameOrg_)
                    if "[_fioDir_]" in paragraph.text:
                        paragraph.text = paragraph.text.replace(
                            "[_fioDir_]", _fioDir_)
                    if "[_adresOrg_]" in paragraph.text:
                        paragraph.text = paragraph.text.replace(
                            "[_adresOrg_]", _adresOrg_)
                    if "[_doljnost_]" in paragraph.text:
                        paragraph.text = paragraph.text.replace(
                            "[_doljnost_]", _doljnost_)
                    if "[_nameOrgSokr_]" in paragraph.text:
                        paragraph.text = paragraph.text.replace(
                            "[_nameOrgSokr_]", _nameOrgSokr_)
    doc.save(path)  # Сохраняем документ после замены текста
