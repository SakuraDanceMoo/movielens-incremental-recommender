"""Render a Markdown course paper as an editable, explicitly formatted DOCX."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Cm, Inches, Pt


def set_font(run,name='宋体',size=12,bold=False):
    run.font.name=name;run.font.size=Pt(size);run.bold=bold
    fonts=run._element.get_or_add_rPr().get_or_add_rFonts()
    fonts.set(qn('w:eastAsia'),name)
    fonts.set(qn('w:ascii'),name)
    fonts.set(qn('w:hAnsi'),name)


def add_text(paragraph,text,name='宋体',size=12,bold=False):
    text=text.replace('`','')
    pieces=re.split(r'(\*\*.*?\*\*)',text)
    for part in pieces:
        if not part:continue
        active_bold=bold or part.startswith('**')
        part=part.strip('*') if part.startswith('**') else part
        cursor=0
        for match in re.finditer(r'\[([^\]]+)\]\((https?://[^)]+)\)',part):
            if match.start()>cursor:set_font(paragraph.add_run(part[cursor:match.start()]),name,size,active_bold)
            link=OxmlElement('w:hyperlink')
            link.set(qn('r:id'),paragraph.part.relate_to(match[2],RT.HYPERLINK,is_external=True))
            r=OxmlElement('w:r');props=OxmlElement('w:rPr')
            fonts=OxmlElement('w:rFonts')
            for key in ['eastAsia','ascii','hAnsi']:fonts.set(qn('w:'+key),name)
            props.append(fonts)
            sz=OxmlElement('w:sz');sz.set(qn('w:val'),str(int(size*2)));props.append(sz)
            color=OxmlElement('w:color');color.set(qn('w:val'),'0563C1');props.append(color)
            underline=OxmlElement('w:u');underline.set(qn('w:val'),'single');props.append(underline)
            r.append(props);t=OxmlElement('w:t');t.text=match[1];r.append(t);link.append(r);paragraph._p.append(link)
            cursor=match.end()
        if cursor<len(part):set_font(paragraph.add_run(part[cursor:]),name,size,active_bold)


def format_paragraph(p,indent=True):
    f=p.paragraph_format
    f.line_spacing=1.5;f.space_before=Pt(0);f.space_after=Pt(4)
    f.first_line_indent=Pt(24) if indent else Pt(0)
    f.widow_control=True


def build(source:Path,output:Path,author:str):
    lines=source.read_text(encoding='utf-8').splitlines()
    doc=Document()
    sec=doc.sections[0]
    sec.page_height=Cm(29.7);sec.page_width=Cm(21)
    sec.top_margin=Cm(2.4);sec.bottom_margin=Cm(2.2)
    sec.left_margin=Cm(2.5);sec.right_margin=Cm(2.5)
    normal=doc.styles['Normal']
    normal.font.name='宋体';normal.font.size=Pt(12)
    normal._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'宋体')
    normal.paragraph_format.line_spacing=1.5
    doc.core_properties.author=author
    doc.core_properties.last_modified_by=author
    doc.core_properties.title=lines[0].lstrip('# ').strip()
    doc.core_properties.subject='大数据技术课程结课报告'
    doc.core_properties.keywords='MovieLens, Transformer, RMSE, 消融实验'
    doc.core_properties.comments=''
    metadata=True
    index=0
    while index<len(lines):
        line=lines[index].strip()
        if not line:index+=1;continue
        if line.startswith('# '):
            p=doc.add_paragraph();format_paragraph(p,False);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            add_text(p,line[2:],size=20,bold=True)
            p.paragraph_format.space_after=Pt(12)
        elif metadata and not line.startswith('【摘要】') and not line.startswith('##'):
            p=doc.add_paragraph();format_paragraph(p,False);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            add_text(p,line,'楷体',10.5)
        elif line=='【摘要】':
            metadata=False
            p=doc.add_paragraph();format_paragraph(p,False);add_text(p,line,bold=True)
        elif line.startswith('关键词'):
            p=doc.add_paragraph();format_paragraph(p,False);add_text(p,line)
        elif re.match(r'^#{2,4} ',line):
            metadata=False
            heading=re.sub(r'^#+\s*','',line)
            p=doc.add_paragraph();format_paragraph(p,False)
            p.paragraph_format.keep_with_next=True;p.paragraph_format.space_before=Pt(10)
            add_text(p,heading,bold=True)
        elif line.startswith('!['):
            match=re.match(r'!\[(.*?)\]\(<?(.*?)>?\)',line)
            if not match:raise ValueError('Malformed image line '+line)
            image=(source.parent/match[2]).resolve()
            p=doc.add_paragraph();format_paragraph(p,False);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next=True
            p.add_run().add_picture(str(image),width=Inches(6.05))
        elif line.startswith('|'):
            rows=[]
            while index<len(lines) and lines[index].strip().startswith('|'):
                cells=[v.strip() for v in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',v) for v in cells):rows.append(cells)
                index+=1
            table=doc.add_table(rows=1,cols=len(rows[0]));table.style='Table Grid'
            table.autofit=True
            for rowidx,values in enumerate(rows):
                row=table.rows[0] if rowidx==0 else table.add_row()
                no_split=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(no_split)
                for cell,text in zip(row.cells,values):
                    p=cell.paragraphs[0];format_paragraph(p,False)
                    p.paragraph_format.space_after=Pt(2)
                    p.paragraph_format.keep_with_next=rowidx<len(rows)-1
                    add_text(p,text,bold=rowidx==0)
                if rowidx==0:
                    repeat=OxmlElement('w:tblHeader');row._tr.get_or_add_trPr().append(repeat)
            doc.add_paragraph().paragraph_format.space_after=Pt(0)
            continue
        else:
            caption=line.startswith(('图 ','表 '))
            p=doc.add_paragraph();format_paragraph(p,not caption)
            if caption:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            if line.startswith('表 '):p.paragraph_format.keep_with_next=True
            add_text(p,line)
        index+=1
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE')
    footer._p.append(field)
    output.parent.mkdir(parents=True,exist_ok=True)
    doc.save(output)
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--author',required=True);args=parser.parse_args()
    build(args.source,args.output,args.author)
