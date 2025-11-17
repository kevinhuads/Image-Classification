from graphviz import Digraph

def build_mlops_overview_graph() -> Digraph:
    """Return a Graphviz Digraph object representing the MLOps overview."""
    dot = Digraph("mlops_overview", format="png")

    # Vertical layout and spacing
    dot.attr(
        rankdir="TB",
        margin="0.1,0.1",   # outer margin (inches)
        pad="0.4",          # padding inside the bounding box
        nodesep="0.4",      # spacing between nodes
        ranksep="0.6",
    )

    # Global style aligned with the dark theme
    dot.attr(
        bgcolor="#0d1b2a",
        fontname="DejaVu Sans",
        fontsize="11",
        fontcolor="#ffffff",
    )

    # Default node style
    dot.attr(
        "node",
        style="filled,rounded",
        shape="box",
        color="#cccccc",
        fillcolor="#0d1b2a",
        fontcolor="#ffffff",
        penwidth="1.2",
    )

    # Default edge style
    dot.attr(
        "edge",
        color="#2a3f5f",
        fontcolor="#ffffff",
        penwidth="1.1",
    )

    # Nodes
    dot.node("data", "Food-101 data\n(images, labels)", shape="folder")
    dot.node("code", "Training code\nsrc/\nconfigs/", shape="component")

    dot.node("github", "GitHub repository\n(code, configs,\nworkflows)", shape="tab")
    dot.node("ci", "CI/CD workflow\n(tests, build, push)", shape="ellipse")
    dot.node("docker", "Docker images\n(app, training)", shape="box")
    dot.node("demo", "Streamlit demo\napp.py", shape="note")
    dot.node("user", "User", shape="oval")

    # Place artifacts (middle) and mlflow (right) on the same horizontal rank
    with dot.subgraph(name="cluster_tracking") as s:
        s.attr(rank="same")
        # Order here matters for left–right placement: artifacts then mlflow
        s.node("artifacts", "Model artifacts\n(checkpoints,\nmetrics, plots)", shape="box3d")
        s.node("mlflow", "MLflow tracking\nserver", shape="cylinder")

    # Data and training loop
    dot.edge("data", "code", label="used by")

    # Straight down to artifacts (middle column)
    dot.edge("code", "artifacts", label="produces")

    # To mlflow on the right; constraint=false so it does not disturb the main vertical flow
    dot.edge("code", "mlflow", label="logs to", constraint="false")

    dot.edge("mlflow", "artifacts", label="stores", constraint="false")

    # Code and GitHub
    dot.edge("code", "github", label="versioned in")

    # GitHub and CI
    dot.edge("github", "ci", label="on push\nor commit")

    # CI and Docker
    dot.edge("ci", "docker", label="build and push")

    # Deployment and usage
    dot.edge("docker", "demo", label="runs")
    dot.edge("artifacts", "demo", label="loads model")
    dot.edge("user", "demo", label="uses")

    return dot

