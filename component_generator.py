import bs4
import os
import re
from bs4 import BeautifulSoup
from typing import Optional


def add_tags(post, user_dict):
    if post["author"] not in user_dict:
        user_dict[post["author"]] = len(user_dict)

    text = str(post)
    user_tag = "[USER" + str(user_dict[post["author"]]) + "]"

    pattern0 = r"(\n\&gt; \*Hello[\S]*)"
    pattern1 = r"(https?://)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*(\.html|\.htm)*"
    pattern2 = r"\&gt;(.*)\n"

    text = text.replace("</claim>", "</claim> ")
    text = text.replace("<claim", " <claim")
    text = text.replace("<premise", " <premise")
    text = text.replace("</premise>", "</premise> ")
    text = re.sub(pattern0, "", text)  # Replace Footnotes
    text = re.sub(pattern1, "[URL]", text)  # Replace [URL]
    text = re.sub(
        pattern2, "[STARTQ]" + r"\1" + " [ENDQ] ", text
    )  # Replace quoted text
    # print(str(text))
    return str(text), user_tag


def get_components(
    component: bs4.BeautifulSoup,
    parent_type: Optional[str] = None,
    parent_id: Optional[str] = None,
    parent_refers: Optional[str] = None,
    parent_rel_type: Optional[str] = None,
):
    """Yields nested components from a parsed component one-by-one. In the form:
    (text, type, id, refers, relation_type)
    text:          The text of the component
    type:          other/claim/premise
    id:            The id of the component, None for Non-Argumentative
    refers:        The ids of the component the current component is related to. None, if it isn't related to any component(separated with _)
    relation_type: The type of relation between current component and refers. None, iff refers is None.
    """

    if str(component).strip() == "":
        yield None

    def chain_yield(comp_type="claim"):
        nonlocal component
        component = str(component)
        parsed_component = BeautifulSoup(component, "xml")

        parent_id = str(parsed_component.find(comp_type)["id"])
        try:
            parent_refers = str(parsed_component.find(comp_type)["ref"])
            parent_rel_type = str(parsed_component.find(comp_type)["rel"])
        except:
            parent_refers = None
            parent_rel_type = None

        for part in parsed_component.find(comp_type).contents:

            if (
                not str(part).strip().startswith("<claim")
                and not str(part).strip().startswith("<premise")
                and not part == parsed_component.find(comp_type).contents[0]
            ):
                parent_ref = parent_id
                parent_id += "Ć"
                parent_rel_type = "cont"

            for _ in get_components(
                part, comp_type, parent_id, parent_refers, parent_rel_type
            ):
                yield _

    if str(component).strip().startswith("<claim"):
        for _ in chain_yield(comp_type="claim"):
            yield _

    elif str(component).strip().startswith("<premise"):
        for _ in chain_yield(comp_type="premise"):
            yield _

    else:
        yield (
            str(component).strip(),
            "other" if parent_type is None else parent_type,
            parent_id,
            parent_refers,
            parent_rel_type,
        )


def generate_components(filename):
    """Yields components from a thread one-by-one. In the form:
    (text, type, id, refers, relation_type)
    text:          The text of the component
    type:          other/claim/premise
    id:            The id of the component, None for Non-Argumentative
    refers:        The ids of the component the current component is related to. None, if it isn't related to any component(separated with _)
    relation_type: The type of relation between current component and refers. None, iff refers is None.
    """

    with open(filename, "r") as g:
        xml_str = g.read()

    parsed_xml = BeautifulSoup(str(BeautifulSoup(xml_str, "lxml")), "xml")

    assert len(re.findall(r"\&\#.*;", str(parsed_xml))) == 0 or re.findall(
        r"\&\#.*;", str(parsed_xml)
    ) == ["&#8217;"], "HTML characters still remaining in XML: " + str(
        re.findall(r"\&\#.*;", str(parsed_xml))
    )

    user_dict = dict()

    yield (
        BeautifulSoup(str(parsed_xml.find("title").find("claim").contents[0]), "lxml")
        .get_text()
        .strip(),
        "claim",
        "title",
        None,
        None,
    )

    for post in [parsed_xml.find("op")] + parsed_xml.find_all("reply"):
        modified_post, user_tag = add_tags(post, user_dict)
        parsed_modified_post = BeautifulSoup(modified_post, "xml")

        try:
            contents = parsed_modified_post.find("op").contents
        except:
            contents = parsed_modified_post.find("reply").contents

        yield (user_tag, "user_tag", None, None, None)

        for component in contents:
            for elem in get_components(component):
                if elem is not None:
                    yield elem


def get_all_threads():
    for t in ["negative", "positive"]:
        root = "AmpersandData/change-my-view-modes/v2.0/" + t + "/"
        for f in os.listdir(root):
            filename = os.path.join(root, f)

            if not (os.path.isfile(filename) and f.endswith(".xml")):
                continue

            for elem in generate_components(filename):
                yield elem
