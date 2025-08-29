# Lesson 6: Reinforcement Learning from Human Feedback (RLHF)

## Overview
This lesson introduces Reinforcement Learning from Human Feedback (RLHF), a technique used to align language models with human preferences and values. We'll explore the theoretical foundations, practical implementation, and applications of RLHF in modern AI systems.

## Learning Objectives
By the end of this lesson, you should be able to:
- Understand the motivation behind RLHF
- Explain the three stages of RLHF training
- Implement a basic RLHF pipeline
- Understand the challenges and limitations of RLHF
- Appreciate the role of RLHF in modern AI alignment

## 1. Introduction to RLHF

### What is RLHF?
Reinforcement Learning from Human Feedback (RLHF) is a technique that uses human preferences to train language models to produce outputs that align with human values and intentions. It has become a crucial component in developing state-of-the-art language models like ChatGPT and Claude.

### Why RLHF?
Traditional language model training with maximum likelihood estimation (MLE) has limitations:
- Models may generate factually incorrect or harmful content
- Models may be verbose, repetitive, or unhelpful
- Models lack alignment with human preferences and values
- Models may exhibit undesirable behaviors like sycophancy or bias

RLHF addresses these issues by incorporating human feedback directly into the training process.

## 2. The Three Stages of RLHF

### Stage 1: Supervised Fine-Tuning (SFT)
Train a model on a dataset of human-written demonstrations to produce helpful responses.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

class SFTDataset(Dataset):
    def __init__(self, prompts, responses, tokenizer, max_length=512):
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        
        # Format as instruction following
        text = f"Human: {prompt}\nAssistant: {response}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def train_sft_model(base_model, train_dataset, val_dataset, config):
    """Train SFT model."""
    model = base_model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        
        print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model
```

### Stage 2: Reward Modeling
Train a reward model that predicts human preferences between different model outputs.

```python
class RewardDataset(Dataset):
    def __init__(self, prompt_response_pairs, human_preferences, tokenizer, max_length=512):
        self.data = prompt_response_pairs
        self.preferences = human_preferences  # 1 if first response preferred, 0 if second
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt, response1, response2 = self.data[idx]
        preference = self.preferences[idx]
        
        # Tokenize both responses
        text1 = f"Human: {prompt}\nAssistant: {response1}"
        text2 = f"Human: {prompt}\nAssistant: {response2}"
        
        encoding1 = self.tokenizer(
            text1,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids_1': encoding1['input_ids'].flatten(),
            'attention_mask_1': encoding1['attention_mask'].flatten(),
            'input_ids_2': encoding2['input_ids'].flatten(),
            'attention_mask_2': encoding2['attention_mask'].flatten(),
            'preference': torch.tensor(preference, dtype=torch.float)
        }

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state of last token as reward features
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden = last_hidden_state[:, -1, :]
        
        reward = self.reward_head(last_token_hidden)
        return reward

def train_reward_model(sft_model, train_dataset, val_dataset, config):
    """Train reward model."""
    model = RewardModel(sft_model)
    model.to(config.device)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids_1 = batch['input_ids_1'].to(config.device)
            attention_mask_1 = batch['attention_mask_1'].to(config.device)
            input_ids_2 = batch['input_ids_2'].to(config.device)
            attention_mask_2 = batch['attention_mask_2'].to(config.device)
            preferences = batch['preference'].to(config.device)
            
            # Get rewards for both responses
            reward1 = model(input_ids_1, attention_mask_1).squeeze()
            reward2 = model(input_ids_2, attention_mask_2).squeeze()
            
            # Compute probability that first response is preferred
            logits = torch.stack([reward1, reward2], dim=1)
            targets = preferences.long()
            
            loss = criterion(logits, targets)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_accuracy = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids_1 = batch['input_ids_1'].to(config.device)
                attention_mask_1 = batch['attention_mask_1'].to(config.device)
                input_ids_2 = batch['input_ids_2'].to(config.device)
                attention_mask_2 = batch['attention_mask_2'].to(config.device)
                preferences = batch['preference'].to(config.device)
                
                reward1 = model(input_ids_1, attention_mask_1).squeeze()
                reward2 = model(input_ids_2, attention_mask_2).squeeze()
                
                # Predict preferences
                predicted = (reward1 > reward2).float()
                val_accuracy += (predicted == preferences).float().mean().item()
        
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = val_accuracy / len(val_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    
    return model
```

### Stage 3: Reinforcement Learning
Use Proximal Policy Optimization (PPO) to fine-tune the model based on the reward model.

```python
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOTrainer:
    def __init__(self, policy_model, reward_model, tokenizer, config):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Freeze reward model
        for param in reward_model.parameters():
            param.requires_grad = False
    
    def compute_rewards(self, responses, prompts):
        """Compute rewards using reward model."""
        rewards = []
        for prompt, response in zip(prompts, responses):
            text = f"Human: {prompt}\nAssistant: {response}"
            encoding = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True
            ).to(self.config.device)
            
            with torch.no_grad():
                reward = self.reward_model(
                    encoding['input_ids'],
                    encoding['attention_mask']
                ).squeeze()
                rewards.append(reward.item())
        
        return torch.tensor(rewards, device=self.config.device)
    
    def compute_advantages(self, rewards, values):
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        # Simplified version - in practice, you'd use value function
        advantages = rewards - values
        return advantages
    
    def ppo_step(self, prompts, old_logprobs, old_values):
        """Perform one PPO training step."""
        # Generate new responses
        new_responses = []
        new_logprobs = []
        new_values = []
        
        for prompt in prompts:
            # Generate response
            inputs = self.tokenizer(f"Human: {prompt}\nAssistant:", return_tensors='pt').to(self.config.device)
            
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    inputs['input_ids'],
                    max_length=self.config.max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            new_responses.append(response)
            
            # Compute log probabilities and values for new responses
            full_text = f"Human: {prompt}\nAssistant: {response}"
            encoding = self.tokenizer(
                full_text,
                return_tensors='pt',
                truncation=True,
                padding=True
            ).to(self.config.device)
            
            with torch.no_grad():
                outputs = self.policy_model(
                    encoding['input_ids'],
                    attention_mask=encoding['attention_mask'],
                    output_hidden_states=True
                )
                
                # Compute log probabilities
                logits = outputs.logits
                logprobs = F.log_softmax(logits, dim=-1)
                # Simplified - in practice, you'd compute logprobs for actual tokens
                
                # Compute values (simplified)
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                value = self.reward_model.reward_head(last_hidden).squeeze()
                
                new_logprobs.append(logprobs.mean())
                new_values.append(value.mean())
        
        # Compute rewards
        rewards = self.compute_rewards(new_responses, prompts)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, torch.stack(new_values))
        
        # Compute PPO loss
        ratios = torch.exp(torch.stack(new_logprobs) - torch.stack(old_logprobs))
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (simplified)
        value_loss = F.mse_loss(torch.stack(new_values), rewards)
        
        # Entropy bonus (encourages exploration)
        # Simplified - in practice, compute entropy of action distribution
        
        total_loss = policy_loss + 0.5 * value_loss  # + entropy_bonus
        
        return total_loss, rewards.mean()

def train_rlhf(sft_model, reward_model, train_prompts, config):
    """Train model using RLHF."""
    policy_model = sft_model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    ppo_trainer = PPOTrainer(policy_model, reward_model, tokenizer, config)
    
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate)
    
    # Initialize with some reference log probabilities
    old_logprobs = [torch.tensor(0.0, device=config.device) for _ in train_prompts]
    old_values = [torch.tensor(0.0, device=config.device) for _ in train_prompts]
    
    for epoch in range(config.epochs):
        # Sample batch of prompts
        batch_indices = torch.randperm(len(train_prompts))[:config.batch_size]
        batch_prompts = [train_prompts[i] for i in batch_indices]
        batch_old_logprobs = [old_logprobs[i] for i in batch_indices]
        batch_old_values = [old_values[i] for i in batch_indices]
        
        # PPO step
        loss, avg_reward = ppo_trainer.ppo_step(
            batch_prompts, 
            batch_old_logprobs, 
            batch_old_values
        )
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Avg Reward: {avg_reward:.4f}, Loss: {loss.item():.4f}")
    
    return policy_model
```

## 3. Practical Implementation Example

### Complete RLHF Pipeline
```python
class RLHFConfig:
    def __init__(self):
        self.model_name = "gpt2"  # Replace with your model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 8
        self.learning_rate = 1e-5
        self.epochs = 3
        self.max_length = 512
        self.clip_epsilon = 0.2

def run_rlhf_pipeline():
    """Run complete RLHF pipeline."""
    config = RLHFConfig()
    
    # Stage 1: Supervised Fine-Tuning
    print("Stage 1: Supervised Fine-Tuning")
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    # Load SFT dataset
    # sft_train_dataset = SFTDataset(...)
    # sft_val_dataset = SFTDataset(...)
    # sft_model = train_sft_model(base_model, sft_train_dataset, sft_val_dataset, config)
    
    # Stage 2: Reward Modeling
    print("Stage 2: Reward Modeling")
    # Load reward dataset
    # reward_train_dataset = RewardDataset(...)
    # reward_val_dataset = RewardDataset(...)
    # reward_model = train_reward_model(sft_model, reward_train_dataset, reward_val_dataset, config)
    
    # Stage 3: Reinforcement Learning
    print("Stage 3: Reinforcement Learning")
    # train_prompts = [...]  # Load training prompts
    # final_model = train_rlhf(sft_model, reward_model, train_prompts, config)
    
    # return final_model

# Example usage
# final_model = run_rlhf_pipeline()
```

## 4. Challenges and Limitations

### Data Quality Issues
- **Human bias**: Human preferences may contain biases
- **Inconsistency**: Humans may give inconsistent preferences
- **Scalability**: Collecting high-quality human feedback is expensive

### Technical Challenges
- **Reward hacking**: Models may exploit reward model weaknesses
- **Distribution shift**: Policy may drift from reward model training distribution
- **Credit assignment**: Difficult to assign credit for long sequences

### Ethical Considerations
- **Value alignment**: Ensuring models align with diverse human values
- **Transparency**: Making the training process interpretable
- **Control**: Maintaining human control over AI systems

## 5. Advanced Topics

### Constitutional AI
An alternative to RLHF that uses AI-generated critiques and revisions instead of human feedback.

```python
def constitutional_ai_step(model, prompt, constitution):
    """Constitutional AI training step."""
    # Generate initial response
    initial_response = generate_response(model, prompt)
    
    # Generate critique using constitution
    critique_prompt = f"Human: {prompt}\nAssistant: {initial_response}\n\nCritique this response according to these principles: {constitution}"
    critique = generate_response(model, critique_prompt)
    
    # Generate revised response
    revision_prompt = f"Human: {prompt}\nAssistant: {initial_response}\n\nCritique: {critique}\n\nPlease revise your response."
    revised_response = generate_response(model, revision_prompt)
    
    return revised_response
```

### DPO (Direct Preference Optimization)
A simpler alternative to PPO that directly optimizes the policy using preference data.

```python
def dpo_loss(policy_logits, reference_logits, preferences):
    """Direct Preference Optimization loss."""
    # Compute log probabilities
    policy_logprobs = F.log_softmax(policy_logits, dim=-1)
    reference_logprobs = F.log_softmax(reference_logits, dim=-1)
    
    # Compute preference probabilities
    policy_prefs = policy_logprobs[:, 0] - policy_logprobs[:, 1]  # First response preferred
    reference_prefs = reference_logprobs[:, 0] - reference_logprobs[:, 1]
    
    # DPO loss
    beta = 0.1  # Temperature parameter
    losses = -F.logsigmoid(beta * (policy_prefs - reference_prefs))
    
    return losses.mean()
```

## 6. Applications and Impact

### Current Applications
- **Chatbots**: Improving conversational quality and helpfulness
- **Content generation**: Ensuring factual accuracy and reducing harmful content
- **Code generation**: Improving code quality and reducing security vulnerabilities
- **Summarization**: Producing more accurate and useful summaries

### Future Directions
- **Automated alignment**: Reducing dependence on human feedback
- **Multi-objective optimization**: Balancing multiple criteria simultaneously
- **Interactive learning**: Continuously learning from user interactions
- **Robustness**: Improving robustness to adversarial inputs

## Summary

In this lesson, we explored Reinforcement Learning from Human Feedback (RLHF):
- The three-stage training process: SFT, reward modeling, and PPO
- Practical implementation of each stage
- Challenges and limitations of RLHF
- Advanced topics like Constitutional AI and DPO
- Applications and future directions

RLHF has become a crucial technique for aligning language models with human preferences and values, enabling the development of more helpful, harmless, and honest AI systems.

## Next Steps

This concludes our core curriculum on transformer models. In the exercises and projects, you'll have opportunities to apply these concepts practically and explore advanced topics in greater depth.