{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNJVPdaWq3dkDPH6xAiSUfG",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/akshayiitr04/satellite-property-valuation/blob/main/data_fetcher.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sv0ZKVIWMuZ6",
        "outputId": "53db47a4-782e-4447-b72f-fe70a18cfa1e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "hello\n"
          ]
        }
      ],
      "source": [
        "print(\"hello\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wxrkUh2iM0HP",
        "outputId": "cb681bb2-bbb9-4cf8-a7ea-d262bd225044"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import mercantile\n",
        "import requests\n",
        "from PIL import Image\n",
        "from io import BytesIO\n",
        "\n",
        "def get_tile_image(z, x, y):\n",
        "    url = f\"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}\"\n",
        "    response = requests.get(url)\n",
        "    if response.status_code == 200:\n",
        "        return Image.open(BytesIO(response.content))\n",
        "    return None\n",
        "\n",
        "\n",
        "def stitch_tiles(lat, lon, zoom=18, tiles_across=3):\n",
        "    center_tile = mercantile.tile(lon, lat, zoom)\n",
        "    half = tiles_across // 2\n",
        "\n",
        "    stitched = Image.new(\"RGB\", (256 * tiles_across, 256 * tiles_across))\n",
        "\n",
        "    for dx in range(-half, half + 1):\n",
        "        for dy in range(-half, half + 1):\n",
        "            x = center_tile.x + dx\n",
        "            y = center_tile.y + dy\n",
        "            img = get_tile_image(zoom, x, y)\n",
        "            if img:\n",
        "                stitched.paste(img, ((dx + half) * 256, (dy + half) * 256))\n",
        "\n",
        "    return stitched\n",
        "\n",
        "def save_satellite_image(image_id, lat, lon,\n",
        "                         base_dir=\"/content/drive/MyDrive/satellite_project/images/raw\",\n",
        "                         zoom=19, tiles_across=3):\n",
        "\n",
        "    os.makedirs(base_dir, exist_ok=True)\n",
        "\n",
        "    save_path = os.path.join(base_dir, f\"{image_id}.jpg\")\n",
        "\n",
        "    # ✅ IMPORTANT: SKIP IF EXISTS\n",
        "    if os.path.exists(save_path):\n",
        "        print(f\"Image {image_id} already exists. Skipping.\")\n",
        "        return\n",
        "\n",
        "    stitched = stitch_tiles(lat, lon, zoom, tiles_across)\n",
        "    resized = stitched.resize((5, 512), Image.LANCZOS)\n",
        "\n",
        "    resized.save(save_path, format=\"JPEG\", quality=85, optimize=True)\n",
        "    print(f\"Saved image {image_id}\")"
      ],
      "metadata": {
        "id": "YDa5232bQEJ-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "nLaRqj_fkV9l"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}