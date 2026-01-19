from utils import get_prompt, create_messages, invoke_llm

with open("scene_graph_to_layout_objects.txt",'r') as file:
    nodes = []
    for line in file:
        nodes.append(line.split('\n')[0])

with open("scene_graph_to_layout_relations.txt",'r') as file:
    edges = []
    for line in file:
        edges.append(line.split('\n')[0])


def create_scene_graph(user_input):
    user_message = get_prompt(
            template_path=r"config/user_prompt_to_scene_graph.yaml",
            role='user',
            user_input=user_input
        )   

    system_message = get_prompt(
        template_path=r"config/user_prompt_to_scene_graph.yaml",
        role='system',
        nodes = nodes,
        edges = edges
        )

    message = create_messages(user_message, system_message)
    llm_response = invoke_llm(message)

    return llm_response

def index_scene_graph(scene_graph):

    objects = scene_graph["objects"]
    relationships = scene_graph["relationships"]
    relationships_index = []

    for triplet in relationships:
        a,b,c = triplet
        relationships_index.append([objects.index(a), b, objects.index(c)])
    
    return {"objects": objects,
            "relationships": relationships_index}


def enrich_regional_prompt(user_input, scene_graph):

    user_message = get_prompt(
            template_path=r"config/regional_prompt_enrichment.yaml",
            role='user'
        )   

    system_message = get_prompt(
        template_path=r"config/regional_prompt_enrichment.yaml",
        role='system',
        user_input=user_input,
        nodes = scene_graph["objects"],
        edges = scene_graph["relationships"]
        )

    message = create_messages(user_message, system_message)
    regional_prompts = invoke_llm(message)


    return list(regional_prompts.values())
