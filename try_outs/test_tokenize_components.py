import os

from tokenize_components import get_tokenized_thread

if __name__ == "__main__":

    distances = []
    dist_relative_to_prev_comment_beginning = []
    title_relations = 0
    cont_relations = 0

    for t in ["negative", "positive"]:
        root = "AmpersandData/change-my-view-modes/v2.0/" + t + "/"
        for f in os.listdir(root):
            filename = os.path.join(root, f)
            if not (os.path.isfile(filename) and f.endswith(".xml")):
                continue
            (
                tokenized_thread,
                begin_positions,
                prev_comment_begin_position,
                ref_n_rel_type,
                _,
                _,
            ) = get_tokenized_thread(filename)
            print("Next tokenized thread: ", f)
            for comp_id in begin_positions:
                ref, rel = ref_n_rel_type[comp_id]
                if ref is None or rel is None:
                    continue
                for ref in ref.split("_"):
                    if ref == "title":
                        title_relations += 1
                        continue
                    if rel == "cont":
                        cont_relations += 1
                        continue
                    distances.append(begin_positions[ref] - begin_positions[comp_id])
                    dist_relative_to_prev_comment_beginning.append(
                        begin_positions[ref] - prev_comment_begin_position[comp_id]
                    )

    print("Title relations: ", title_relations)
    print("Continue Relations: ", cont_relations)
    print("Relative Distances: ", distances)
    print(
        "Distances w.r.t. beginning of previous comment of the current component's commment: ",
        dist_relative_to_prev_comment_beginning,
    )
