from collections import defaultdict
import os, random, math
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

sns.set_theme(
    style="darkgrid",           
    rc={
        "figure.facecolor": "#0d1b2a",
        "axes.facecolor":   "#0d1b2a",
        "axes.edgecolor":   "#cccccc",
        "grid.color":       "#2a3f5f",
        "axes.labelcolor":  "#ffffff",
        "text.color":       "#ffffff",
        "xtick.color":      "#ffffff",
        "ytick.color":      "#ffffff",
    },
    palette="deep"               
)

def sample_distinct_classes(train_list, n, data_dir):
    by_class = defaultdict(list)
    for rel in train_list:
        cls = os.path.normpath(rel).split(os.sep)[0]
        by_class[cls].append(rel)
    classes = random.sample(list(by_class.keys()), min(n, len(by_class)))
    return [os.path.join(data_dir, "images", random.choice(by_class[c])) for c in classes]

def show_grid(image_paths, ncols=5, figsize=(12, 14), titles=None,
              title_loc="center", title_fontsize=10, title_color="white",
              title_pad=6, facecolor="black"):
    """Display a grid of images."""
    n = len(image_paths)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    if titles is None:
        titles = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    for i, p in enumerate(image_paths):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.set_facecolor(facecolor)
        img = Image.open(p).convert("RGB")
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(titles[i], fontsize=title_fontsize, color=title_color,
                     pad=title_pad, loc=title_loc)
    # hide unused cells
    total = nrows * ncols
    for j in range(n, total):
        ax = plt.subplot(nrows, ncols, j + 1)
        ax.axis("off")
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    plt.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
    plt.show()

def plot_image_size_stats(image_dir, train_list, val_list, sample_size = 0.1, seed = 3):
    paths = np.array(list(train_list) + list(val_list))

    np.random.seed(seed)
    n = int(np.ceil(sample_size * len(paths)))
    paths = paths[np.random.choice(len(paths), size=n, replace=False)]

    sizes = np.array([Image.open(os.path.join(image_dir, p)).size for p in paths])

    widths, heights = sizes.T
    aspect_ratios = widths / heights

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    for ax, arr, title, xlabel in [
        (axs[0, 0], widths, "Width distribution", "width"),
        (axs[0, 1], heights, "Height distribution", "height"),
        (axs[1, 0], aspect_ratios, "Aspect ratio (w/h)", "aspect ratio"),
    ]:
        counts, bins, _ = ax.hist(arr, bins=20, weights=np.full(arr.shape, 100 / len(arr)))
        ax.set(title=title, xlabel=xlabel, ylabel="Percentage")
        ax.set_ylim(0, counts.max() * 1.15)
        top = counts.argsort()[-5:]
        xs = (bins[top] + bins[top + 1]) / 2
        for x, c in zip(xs, counts[top]):
            ax.annotate(f"{c:.1f}%", (x, c), xytext=(0, 6),
                        textcoords="offset points", ha="center")

    pairs, freqs = zip(*Counter(map(tuple, sizes)).most_common(10))
    perc = np.array(freqs) * 100 / len(sizes)

    ax = axs[1, 1]
    x = np.arange(len(pairs))
    ax.bar(x, perc)
    ax.set_xticks(x, [f"{w}×{h}" for w, h in pairs], rotation=45, ha="right")
    ax.set(ylabel="Percentage", title="Top 10 (width, height) duos")
    ax.set_ylim(0, perc.max() * 1.15)
    for i, y in enumerate(perc):
        ax.annotate(f"{y:.1f}%", (x[i], y), xytext=(0, 6),
                    textcoords="offset points", ha="center")

    plt.tight_layout()
    plt.show()
    
# ----------------------
# Transforms & normalization helpers
# ----------------------

def get_transforms(img_size: int = 224):
    pil_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    imgfolder_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])
    imagenet_norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return pil_transform, imgfolder_transform, imagenet_norm

# ----------------------
# Model & embedding helpers
# ----------------------

def build_model(variant: str = "sequential", device: Optional[torch.device] = None,
                weights=models.ResNet50_Weights.DEFAULT) -> torch.nn.Module:
    """Return a ResNet50 backbone suitable for producing 2048-d embeddings.

    - variant='sequential' returns a Sequential of all children except the final fc layer
      which produces shape (N,2048,1,1) and requires flattening.
    - variant='identity' replaces the fc with Identity and returns outputs of shape (N,2048).

    The returned model is moved to `device` and set to eval().
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if variant == "sequential":
        m = models.resnet50(weights=weights)
        m = torch.nn.Sequential(*list(m.children())[:-1])
        m = m.to(device).eval()
        return m
    else:
        m = models.resnet50(weights=weights)
        m.fc = torch.nn.Identity()
        m = m.to(device).eval()
        return m


@torch.no_grad()
def embed_batch_from_tensor(imgs: torch.Tensor,
                            model: torch.nn.Module,
                            imagenet_norm: transforms.Normalize,
                            model_variant: str = "sequential",
                            device: Optional[torch.device] = None,
                            normalize_embeddings: bool = False) -> np.ndarray:
    """Embed a batch of images (float tensor in [0,1]) to a (B,2048) numpy array.

    Args:
        imgs: Tensor shape (B,3,H,W) with values in [0,1].
        model: backbone returned by build_model.
        imagenet_norm: normalization transform to apply before model.
        model_variant: 'sequential' or 'identity' (controls flattening).
        device: torch device (if None, auto-selected).
        normalize_embeddings: if True, L2-normalize output vectors.

    Returns:
        numpy array (B,2048)
    """
    if device is None:
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgs = imagenet_norm(imgs).to(device, non_blocking=True)
    out = model(imgs)
    if model_variant == "sequential":
        # (B,2048,1,1) -> (B,2048)
        out = out.view(out.size(0), -1)
    # else identity variant already flattened
    feats = out
    if normalize_embeddings:
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()

# ----------------------
# Dataset helpers
# ----------------------

def make_imagefolder(image_dir: str, imgfolder_transform: transforms.Compose) -> ImageFolder:
    ds = ImageFolder(root=image_dir, transform=imgfolder_transform)
    return ds


def sample_subset(ds: ImageFolder, num_classes: int = 10, per_class: int = 200, seed: int = 3) -> Tuple[List[int], List[str]]:
    rng = random.Random(seed)
    classes = ds.classes
    chosen = rng.sample(classes, num_classes)
    cls2idx = {c:i for i,c in enumerate(classes)}
    indices = []
    for c in chosen:
        idx = cls2idx[c]
        all_idxs = [i for i,(_, lab) in enumerate(ds.samples) if lab == idx]
        sel = rng.sample(all_idxs, min(per_class, len(all_idxs)))
        indices.extend(sel)
    return indices, chosen


def build_full_loader(ds: ImageFolder, subsample_n: Optional[int] = 20000,
                      batch_size: int = 256, num_workers: int = 4, seed: int = 3) -> Tuple[DataLoader, object]:
    n_total = len(ds)
    if subsample_n is None or subsample_n >= n_total:
        chosen_ds = ds
    else:
        perm = np.random.RandomState(seed).permutation(n_total)[:subsample_n]
        chosen_ds = Subset(ds, perm.tolist())
    loader = DataLoader(chosen_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, chosen_ds

# ----------------------
# Embedding extraction flows
# ----------------------

def extract_embeddings_sample(ds: ImageFolder,
                              model: torch.nn.Module,
                              pil_transform: transforms.Compose,
                              indices: List[int],
                              classes: List[str],
                              model_variant: str = "sequential",
                              device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract embeddings by reading image files individually (reproduce original sample-mode behaviour).

    Returns X (N,2048), y (N,), sample_classes (list of chosen class names in order used to remap labels).
    """
    X_list, y_list = [], []
    sample_classes = []
    # If user provided classes argument, we assume indices were sampled from that same classes list
    # but to be consistent we allow classes param to be ds.classes from caller.
    for idx in indices:
        path, lab = ds.samples[idx]
        img = Image.open(path).convert("RGB")
        x = pil_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            if model_variant == "sequential":
                out = out.squeeze()
                feat = out.cpu().numpy().ravel()
            else:
                feat = out.squeeze().cpu().numpy().ravel()
        X_list.append(feat)
        # Note: caller should pass the sampled class list ordering if they need remapped labels.
        y_list.append(lab)
    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y, []


def extract_embeddings_from_loader(loader: DataLoader,
                                   model: torch.nn.Module,
                                   imagenet_norm: transforms.Normalize,
                                   model_variant: str = "sequential",
                                   device: Optional[torch.device] = None,
                                   normalize_embeddings: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    feats_parts, labels_parts = [], []
    for imgs, labs in loader:
        feats = embed_batch_from_tensor(imgs, model, imagenet_norm, model_variant=model_variant,
                                        device=device, normalize_embeddings=normalize_embeddings)
        feats_parts.append(feats)
        labels_parts.append(labs.numpy())
    X = np.vstack(feats_parts)
    y = np.concatenate(labels_parts)
    return X, y

# ----------------------
# Projection helpers
# ----------------------

def compute_tsne_direct(X: np.ndarray, n_components: int = 2, perplexity: int = 30,
                        random_state: int = 0, init: str = "pca", max_iter: int = 1000) -> np.ndarray:
    tsne_direct = TSNE(n_components=n_components, init=init, perplexity=perplexity,
                       random_state=random_state, max_iter=max_iter)
    Z = tsne_direct.fit_transform(X)
    return Z


def compute_pca_then_tsne(X: np.ndarray, pca_dim: int = 50, tsne_perplexity: int = 30,
                          tsne_iter: int = 1000, random_seed: int = 3) -> Tuple[np.ndarray, PCA]:
    if X.shape[0] > 50000:
        pca = IncrementalPCA(n_components=pca_dim)
        parts = []
        chunk = 5000
        for i in range(0, X.shape[0], chunk):
            part = X[i:i+chunk]
            if i == 0:
                pca.partial_fit(part)
            parts.append(pca.transform(part))
        X_pca = np.vstack(parts)
    else:
        pca = PCA(n_components=min(pca_dim, X.shape[1]), random_state=random_seed)
        X_pca = pca.fit_transform(X)
    tsne = TSNE(n_components=2, init="pca", perplexity=tsne_perplexity,
                learning_rate="auto", random_state=random_seed, max_iter=tsne_iter)
    Z = tsne.fit_transform(X_pca)
    return Z, pca

# ----------------------
# Clustering diagnostics
# ----------------------

def run_kmeans_metrics(feat_for_clustering: np.ndarray, y: np.ndarray,
                       k_clusters: int = 101, kmeans_batch: int = 1024, random_seed: int = 3) -> Tuple[np.ndarray, float, float]:
    kmeans = MiniBatchKMeans(n_clusters=k_clusters, random_state=random_seed, batch_size=kmeans_batch)
    clusters = kmeans.fit_predict(feat_for_clustering)
    ari = adjusted_rand_score(y, clusters)
    nmi = normalized_mutual_info_score(y, clusters)
    return clusters, ari, nmi

# ----------------------
# Intra-class spread
# ----------------------

def compute_spread_from_projection(proj: np.ndarray, y: np.ndarray, classes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df2 = pd.DataFrame({"x": proj[:,0], "y": proj[:,1], "ytrue": y})
    spread = (df2.groupby("ytrue")[["x","y"]]
              .agg("std")
              .pow(2)
              .sum(axis=1)
              .pow(0.5)
              .rename("spread")
              .reset_index())
    spread["class"] = spread["ytrue"].apply(lambda i: classes[int(i)])
    hardest = spread.sort_values("spread", ascending=False).head(15)[["class","spread"]]
    easiest = spread.sort_values("spread", ascending=True).head(15)[["class","spread"]]
    return spread, hardest, easiest

# ----------------------
# Centroids and dendrogram utilities
# ----------------------

def compute_class_centroids(source_ds: ImageFolder,
                            classes: List[str],
                            embed_batch_fn,
                            imagenet_norm: transforms.Normalize,
                            centroid_samples_per_class: int = 64,
                            batch_size: int = 256,
                            model_variant: str = "sequential",
                            normalize_embeddings: bool = False) -> np.ndarray:
    """Compute per-class centroids (L2-normalized) using up to centroid_samples_per_class images per class.

    embed_batch_fn: function compatible with embed_batch_from_tensor signature.
    """
    idx_by_class = defaultdict(list)
    for i, (_, lab) in enumerate(source_ds.samples):
        idx_by_class[classes[lab]].append(i)

    centroids = []
    for c_idx, cls in enumerate(classes):
        idxs = idx_by_class[cls][:centroid_samples_per_class]
        if not idxs:
            centroids.append(np.zeros(2048, dtype=np.float32))
            continue
        feats_local = []
        for i in range(0, len(idxs), batch_size):
            batch_idxs = idxs[i:i+batch_size]
            imgs = [imagenet_norm(source_ds[j][0]) for j in batch_idxs]
            imgs = torch.stack(imgs, 0)
            feats_local.append(embed_batch_fn(imgs))
        feats_local = np.vstack(feats_local)
        centroid = feats_local.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids.append(centroid)
    return np.vstack(centroids)


def compute_linkage_from_centroids(centroids: np.ndarray, method: str = "average") -> np.ndarray:
    # cosine distance matrix (condensed form for linkage)
    D = 1.0 - (centroids @ centroids.T)
    condensed = D[np.triu_indices_from(D, k=1)]
    zlink = linkage(condensed, method=method)
    return zlink


def assign_flat_clusters(zlink: np.ndarray, n_dendro_clusters: int) -> np.ndarray:
    return fcluster(zlink, t=n_dendro_clusters, criterion="maxclust")


def build_class_color_mapping(classes: List[str], cluster_assign: np.ndarray,
                              n_dendro_clusters: int) -> Tuple[Dict[str,str], Dict[int,str], List[str]]:
    n = len(classes)
    if n_dendro_clusters <= 20:
        palette = sns.color_palette("tab20", n_dendro_clusters)
    else:
        palette = sns.color_palette("hsv", n_dendro_clusters)
    hex_palette = [mcolors.to_hex(c) for c in palette]
    class_to_color = {
        classes[i]: hex_palette[(int(cluster_assign[i]) - 1) % len(hex_palette)]
        for i in range(n)
    }
    leaf_cluster = {i: int(cluster_assign[i]) for i in range(n)}
    return class_to_color, leaf_cluster, hex_palette


def make_link_color_func(zlink: np.ndarray, n: int, leaf_cluster: Dict[int,int], hex_palette: List[str],
                         fallback_hex: str = "#808080"):
    # build node->leaves mapping
    node_to_leaves = {}
    for row_idx in range(zlink.shape[0]):
        left = int(zlink[row_idx, 0])
        right = int(zlink[row_idx, 1])
        leaves = []
        for child in (left, right):
            if child < n:
                leaves.append(child)
            else:
                leaves.extend(node_to_leaves[child - n])
        node_to_leaves[row_idx] = leaves

    def link_color_func(link_id):
        lid = int(link_id)
        if lid < n:
            return fallback_hex
        row = lid - n
        leaves = node_to_leaves.get(row, [])
        if not leaves:
            return fallback_hex
        clusters = [leaf_cluster[leaf] for leaf in leaves]
        first = clusters[0]
        if all(c == first for c in clusters):
            return hex_palette[(first - 1) % len(hex_palette)]
        else:
            return fallback_hex

    return link_color_func


def plot_vertical_dendrogram(zlink: np.ndarray, classes: List[str], class_to_color: Dict[str,str],
                             link_color_func, figsize: Tuple[int,int] = (10,20), title: str = "class dendrogram (embedding centroid similarity)"):
    n = len(classes)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    dendro = dendrogram(
        zlink,
        labels=classes,
        orientation="right",
        link_color_func=link_color_func,
        color_threshold=0,
        leaf_font_size=10,
        ax=ax
    )
    for t in ax.get_yticklabels():
        lbl = t.get_text()
        if lbl in class_to_color:
            t.set_color(class_to_color[lbl])
    plt.title(title, pad=16)
    plt.tight_layout()
    plt.show()

# End of module


def run_embedding_pipeline(
    mode,
    ds,
    model: torch.nn.Module,
    pil_transform: transforms.Compose,
    imagenet_norm: transforms.Normalize,
    device: Optional[torch.device] = None,
    *,
    # sampling / full options
    sample_num_classes: int = 10,
    sample_per_class: int = 200,
    subsample_n: Optional[int] = 20000,
    batch_size: int = 256,
    num_workers: int = 4,
    random_seed: int = 3,
    # projection choices
    pca_before_tsne: bool = False,
    pca_dim: int = 50,
    tsne_perplexity: int = 30,
    tsne_iter: int = 1000,
):
    """Run the embedding extraction flow for either 'sample' or 'full' modes.

    Returns a dict with keys:
      - X: (N,2048) embeddings
      - y: (N,) integer labels (indices into ds.classes)
      - labels: list of human-readable labels (only populated for 'sample')
      - chosen_ds: the dataset used for full-mode (Subset or ImageFolder)
      - chosen_classes: list of sampled class names for sample-mode
      - Z: 2D projection (t-SNE) if computed
      - pca: PCA object if pca_before_tsne used, else None

    The function centralises the previous `if mode==...` branching so callers only need to set `mode`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    out = {
        "X": None,
        "y": None,
        "labels": None,
        "chosen_ds": None,
        "chosen_classes": None,
        "Z": None,
        "pca": None,
    }

    if mode == "sample":
        indices, chosen_classes = sample_subset(ds, num_classes=sample_num_classes,
                                                per_class=sample_per_class, seed=random_seed)
        X, y, _ = extract_embeddings_sample(ds=ds, model=model, pil_transform=pil_transform,
                                            indices=indices, classes=ds.classes,
                                            model_variant=("sequential" if isinstance(model, torch.nn.Sequential) else "identity"),
                                            device=device)

        cls2pos = {c:i for i,c in enumerate(chosen_classes)}
        y_remap = np.array([cls2pos[ds.classes[int(l)]] for l in y])
        labels = [chosen_classes[int(i)] for i in y_remap]
        out.update({"X": X, "y": y_remap, "labels": labels, "chosen_ds": ds, "chosen_classes": chosen_classes})
    else:
        loader, chosen_ds = build_full_loader(ds, subsample_n=subsample_n,
                                             batch_size=batch_size, num_workers=num_workers, seed=random_seed)
        X, y = extract_embeddings_from_loader(loader, model, imagenet_norm,
                                              model_variant=("sequential" if isinstance(model, torch.nn.Sequential) else "identity"),
                                              device=device, normalize_embeddings=False)
        out.update({"X": X, "y": y, "labels": [ds.classes[int(i)] for i in y], "chosen_ds": chosen_ds, "chosen_classes": ds.classes})

    # projection
    if not pca_before_tsne:
        try:
            Z = compute_tsne_direct(out["X"], perplexity=tsne_perplexity, random_state=random_seed, max_iter=tsne_iter)
            out["Z"] = Z
        except Exception:
            out["Z"] = None
    else:
        try:
            Z, pca = compute_pca_then_tsne(out["X"], pca_dim=pca_dim, tsne_perplexity=tsne_perplexity,
                                           tsne_iter=tsne_iter, random_seed=random_seed)
            out["Z"] = Z
            out["pca"] = pca
        except Exception:
            out["Z"] = None
            out["pca"] = None

    return out
