# Sistema de Detecção e Contagem de Tênis com YOLOv8

## Desafio Técnico – Dev Backend com foco em IA

Este projeto é a minha solução para o Desafio Técnico proposto, que consiste em criar um sistema de visão computacional para detectar, rastrear e contar tênis em uma área definida, simulando um processo de coleta em um ambiente com visão limitada.

O sistema foi desenvolvido em Python, utilizando um modelo de detecção de objetos **YOLOv8** customizado, e a biblioteca **OpenCV** para o processamento de vídeo e a lógica de rastreamento.

---

### 🎥 Vídeo de Apresentação
https://drive.google.com/file/d/1qNgVfnb4zVxoT2aygI4ZWrR46_ByN7RU/view?usp=sharing

---

### ✨ Funcionalidades Principais

* **Treinamento de Modelo Customizado:** Script (`train.py`) para treinar um modelo YOLOv8 a partir de um dataset de imagens de tênis.
* **Detecção de Objetos:** Utiliza o modelo treinado para detectar tênis em um stream de vídeo.
* **Região de Interesse (ROI):** Permite que o usuário defina interativamente a área de monitoramento no início da execução.
* **Rastreamento de Objetos (Tracking):** Implementa um rastreador de centroides para atribuir um ID único a cada tênis, permitindo o acompanhamento individual ao longo do tempo.
* **Lógica de Coleta:** Detecta quando um tênis rastreado desaparece da ROI por um período determinado, registrando-o como "coletado".
* **Contagem em Tempo Real:** Exibe na tela os contadores atualizados de `Total Detectado`, `Coletados` e `Restantes na Cena`.
* **Relatório Final:** Gera um relatório consolidado no terminal ao final da execução.

---

### 🛠️ Tecnologias Utilizadas

* **Python 3.10+**
* **PyTorch:** Framework base para o funcionamento do YOLOv8.
* **Ultralytics YOLOv8:** Para treinamento e inferência do modelo de detecção de objetos.
* **OpenCV-Python:** Para manipulação de vídeo, definição da ROI e visualizações.
* **NumPy:** Para cálculos matriciais e manipulação de arrays.

---

### ⚙️ Configuração e Instalação

**1. Clone o Repositório**
```bash
git clone https://github.com/saraivajv/shoes-processing
cd shoes-processing
```

**2. Crie e Ative um Ambiente Virtual (Recomendado)**
```bash
# Para Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Para Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Instale as Dependências**
As dependências principais podem ser instaladas via pip.
```bash
pip install ultralytics opencv-python numpy
```
*(Nota: A biblioteca `ultralytics` instalará o `PyTorch` e outras dependências necessárias automaticamente.)*

**4. Estrutura de Arquivos**
Certifique-se de que o modelo treinado (`best.pt`) e o vídeo de simulação (`video_simulacao.mp4`) estejam nos caminhos corretos, conforme especificado no topo do script `main.py`.

---

### ▶️ Como Executar

O projeto é dividido em duas partes principais: treinamento (opcional, pois o modelo já está treinado) e execução.

**1. Treinamento (Opcional)**
Para treinar o modelo do zero, você precisará de um dataset no formato YOLO e um arquivo `data.yaml` configurado.
```bash
python train.py
```
O melhor modelo será salvo no diretório `runs/detect/yolov8_tenis_detector/weights/best.pt`.

**2. Execução Principal (Detecção e Contagem)**
Este é o script principal que executa a solução completa.
```bash
python main.py
```
**Instruções de Interação:**
1.  Ao executar, uma janela com o primeiro frame do vídeo aparecerá.
2.  **Clique e arraste o mouse** para desenhar um retângulo sobre a área que deseja monitorar (a ROI).
3.  Pressione **ENTER** ou **ESPAÇO** para confirmar.
4.  A detecção e o rastreamento começarão, e os resultados serão exibidos em tempo real. Pressione **'q'** para sair.

---

### 🧠 Lógica e Decisões Técnicas

A solução foi construída sobre duas camadas principais: detecção e rastreamento.

1.  **Detector (YOLOv8):** A escolha do YOLOv8 se deu por sua alta performance e facilidade de customização. O *transfer learning* a partir do modelo `yolov8n` permitiu alcançar bons resultados com um tempo de treinamento relativamente curto.

2.  **Rastreador (Tracker):** Foi implementado um rastreador de centroides simples, baseado na **distância euclidiana**. Ele associa detecções em frames consecutivos ao ID existente mais próximo, o que é eficiente para objetos que não se movem de forma extremamente errática.

3.  **Robustez e Refinamentos:** Para lidar com os desafios de um cenário real (oclusões, detecções instáveis), foram implementados os seguintes refinamentos:
    * **Período de Inicialização:** Os primeiros `80` frames são usados para o sistema "aprender" a cena e estabilizar a contagem inicial de objetos, evitando que "piscadas" na detecção sejam contadas como coletas.
    * **Filtro de Ruído:** Detecções com uma área em pixels muito pequena são descartadas para evitar que ruídos do modelo criem "tênis fantasmas" e inflem a contagem.
    * **Ajuste de NMS (IoU Threshold):** O limiar de Interseção sobre União foi ajustado para `0.4` para que o detector lide melhor com tênis que estão muito próximos uns dos outros.

---

### 🚀 Melhorias Futuras

* **Aprimoramento do Modelo:** Treinar com um dataset mais diverso (mais fundos, iluminações e oclusões) e por mais épocas para melhorar a detecção de objetos em condições desafiadoras.
* **Implementar um Rastreador Avançado:** Utilizar algoritmos como **SORT** ou **DeepSORT**, que são mais robustos a oclusões longas e trocas de ID.
* **Diferenciação de Pares:** Evoluir o modelo para não apenas detectar "tênis", mas também para classificar se é um pé direito ou esquerdo, permitindo a contagem de pares.
* **API Backend:** Expor a funcionalidade do sistema através de uma API REST, onde um cliente poderia enviar um vídeo e receber o relatório final como um JSON.

---
**Autor:** João Victor G. de A. Saraiva
