# Fine-Tuning-LLM

# LLM Fine-tuning

LLM Fine-tuning involves the additional training of a pre-existing model, which has previously acquired patterns and features from an extensive dataset, using a smaller, domain-specific dataset. In the context of "LLM Fine-Tuning," LLM denotes a "Large Language Model," such as the GPT series by OpenAI. This approach holds significance as training a large language model from the ground up is highly resource-intensive in terms of both computational power and time. Utilizing the existing knowledge embedded in the pre-trained model allows for achieving high performance on specific tasks with substantially reduced data and computational requirements.

## Key Steps Involved in LLM Fine-tuning:

1. **Select a Pre-trained Model**: Carefully choose a base pre-trained model that aligns with the desired architecture and functionalities. Pre-trained models are generic purpose models that have been trained on a large corpus of unlabeled data.

2. **Gather Relevant Dataset**: Collect a dataset that is relevant to the task at hand. The dataset should be labeled or structured in a way that the model can learn from it.

3. **Preprocess Dataset**: Prepare the dataset for fine-tuning by cleaning it, splitting it into training, validation, and test sets, and ensuring it's compatible with the model on which we want to fine-tune.

4. **Fine-tuning**: After selecting a pre-trained model, fine-tune it on the preprocessed relevant dataset, which is more specific to the task at hand. This dataset might be related to a particular domain or application, allowing the model to adapt and specialize for that context.

5. **Task-specific Adaptation**: During fine-tuning, adjust the model's parameters based on the new dataset, helping it better understand and generate content relevant to the specific task. This process retains the general language knowledge gained during pre-training while tailoring the model to the nuances of the target domain.

# Fine-tuning Methods

Fine-tuning a Large Language Model (LLM) involves a supervised learning process. In this method, a dataset comprising labeled examples is utilized to adjust the model’s weights, enhancing its proficiency in specific tasks. Now, let’s delve into some noteworthy techniques employed in the fine-tuning process.

## Full Fine Tuning (Instruction fine-tuning):

Instruction fine-tuning is a strategy to enhance a model’s performance across various tasks by training it on examples that guide its responses to queries. The choice of the dataset is crucial and tailored to the specific task, such as summarization or translation. This approach, known as full fine-tuning, updates all model weights, creating a new version with improved capabilities. However, it demands sufficient memory and computational resources, similar to pre-training, to handle the storage and processing of gradients, optimizers, and other components during training.

## Parameter Efficient Fine-Tuning (PEFT):

Parameter Efficient Fine-Tuning (PEFT) is a form of instruction fine-tuning that is much more efficient than full fine-tuning. Training a language model, especially for full LLM fine-tuning, demands significant computational resources. Memory allocation is not only required for storing the model but also for essential parameters during training, presenting a challenge for simple hardware. PEFT addresses this by updating only a subset of parameters, effectively “freezing” the rest. This reduces the number of trainable parameters, making memory requirements more manageable and preventing catastrophic forgetting. Unlike full fine-tuning, PEFT maintains the original LLM weights, avoiding the loss of previously learned information. This approach proves beneficial for handling storage issues when fine-tuning for multiple tasks. There are various ways of achieving Parameter efficient fine-tuning. Low-Rank Adaptation LoRA & QLoRA are the most widely used and effective.

# LoRA and QLoRA

## LoRA (Low-Rank Adaptation):

LoRA is an improved fine-tuning method where instead of fine-tuning all the weights that constitute the weight matrix of the pre-trained large language model, two smaller matrices that approximate this larger matrix are fine-tuned. These matrices constitute the LoRA adapter. This fine-tuned adapter is then loaded into the pre-trained model and used for inference.

After LoRA fine-tuning for a specific task or use case, the outcome is an unchanged original LLM and the emergence of a considerably smaller “LoRA adapter,” often representing a single-digit percentage of the original LLM size (in MBs rather than GBs).

During inference, the LoRA adapter must be combined with its original LLM. The advantage lies in the ability of many LoRA adapters to reuse the original LLM, thereby reducing overall memory requirements when handling multiple tasks and use cases.

## Quantized LoRA (QLoRA):

QLoRA represents a more memory-efficient iteration of LoRA. QLoRA takes LoRA a step further by also quantizing the weights of the LoRA adapters (smaller matrices) to lower precision (e.g., 4-bit instead of 8-bit). This further reduces the memory footprint and storage requirements. In QLoRA, the pre-trained model is loaded into GPU memory with quantized 4-bit weights, in contrast to the 8-bit used in LoRA. Despite this reduction in bit precision, QLoRA maintains a comparable level of effectiveness to LoRA.

[Link to the article](https://dassum.medium.com/fine-tune-large-language-model-llm-on-a-custom-dataset-with-qlora-fb60abdeba07#:~:text=Fine%2Dtuning%20a%20Large%20Language,in%20the%20fine%2Dtuning%20process)

