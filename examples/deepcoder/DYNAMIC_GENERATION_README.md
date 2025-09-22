# Dynamic Problem Generation for DeepCoder Training

This feature enables the DeepCoder PPO trainer to dynamically generate new programming problems using GPT-4o after a specified number of training steps. The generated problems are based on the model's recent outputs, allowing for adaptive curriculum learning.

## How It Works

1. **Initial Training Phase**: The trainer uses the original DeepCoder dataset for the first N steps (default: 50 steps).

2. **Transition to Dynamic Generation**: After reaching the threshold, the trainer:
   - Extracts a sample output from the current training batch
   - Makes multiple API calls to GPT-4o (one per problem) to generate programming problems
   - Each call generates a single, high-quality problem tailored to the model's current ability
   - Creates a new dataloader with these problems

3. **Ongoing Dynamic Generation**: 
   - New problems are generated at the end of each epoch (configurable frequency)
   - Each generation uses the most recent model output as context
   - This creates an adaptive curriculum that evolves with the model

## Configuration

Add these settings to your trainer config:

```yaml
trainer:
  # Enable dynamic generation after 50 steps
  dynamic_generation_threshold: 50
  
  # Generate new problems every 20 steps
  dynamic_generation_frequency: 20
  
  # Number of problems to generate each time (generated one at a time)
  problems_per_generation: 8
  
  # Set your OpenAI API key
  openai_api_key: "your-openai-api-key-here"
  
  # ... other trainer settings
```

## Usage Example

```python
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer
from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.environments.base.single_turn_env import SingleTurnEnvironment

# Initialize trainer with dynamic generation config
trainer = AgentPPOTrainer(
    config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    reward_fn=reward_fn,
    val_reward_fn=val_reward_fn,
    env_class=SingleTurnEnvironment,
    agent_class=CompetitionCodingAgent
)

# Start training - will automatically switch to dynamic generation after threshold
trainer.fit_agent()
```

## Generated Problem Format

The system expects GPT-4o to return problems in this format:

```
<problem>
Write a function to find the longest common subsequence of two strings...

-----Input-----
The first line contains two strings s1 and s2...

-----Output-----
Print the length of the longest common subsequence...

-----Examples-----
Input: abcde ace
Output: 3
</problem>

<tests>
[
  {"input": "abcde\nace", "output": "3", "testtype": "stdin_stdout"},
  {"input": "abc\nabc", "output": "3", "testtype": "stdin_stdout"}
]
</tests>

<metadata>
{
  "difficulty": "Medium",
  "type": "Dynamic Programming",
  "func_name": null
}
</metadata>
```

## Features

### Adaptive Difficulty
- Problems are generated based on current model performance
- Difficulty levels: Easy, Medium, Hard
- Algorithm types: Array, String, Graph, Dynamic Programming, etc.

### Comprehensive Logging
- Tracks number of problems generated
- Monitors difficulty distribution
- Logs problem type distribution
- Saves generated problems for analysis

### Error Handling
- Validates generated problems before use
- Falls back to original dataset if generation fails
- Robust error handling for API calls

### Problem Validation
- Ensures all required fields are present
- Validates test case format
- Checks JSON structure integrity

## Output Files

Generated problems are saved to:
- `{checkpoint_dir}/generated_problems/problems_step_{step}.json`

## Monitoring

The system logs comprehensive metrics:
- `dynamic_generation/problems_generated`: Number of problems generated
- `dynamic_generation/total_generated`: Total problems generated so far
- `dynamic_generation/difficulty/{level}`: Distribution by difficulty
- `dynamic_generation/type/{type}`: Distribution by algorithm type

## Benefits

- **Adaptive Curriculum**: Problems evolve with model capabilities
- **Diverse Problem Types**: Covers various algorithm categories
- **Continuous Learning**: Model faces new challenges throughout training
- **Quality Control**: Validates generated problems before use
- **Comprehensive Monitoring**: Tracks generation effectiveness

## Requirements

- OpenAI API key with GPT-4o access
- Internet connection for API calls
- Sufficient disk space for generated problem storage

## Troubleshooting

1. **No problems generated**: Check OpenAI API key and internet connection
2. **Invalid problem format**: Check GPT-4o output parsing
3. **Generation failures**: System will fallback to original dataset
4. **High API costs**: Adjust `problems_per_generation` and `dynamic_generation_frequency` (note: problems are generated one at a time)

## Example Training Script

```bash
# Train with dynamic generation
python -m examples.deepcoder.train_deepcoder \
    --config-path examples/deepcoder/dynamic_deepcoder_config.yaml \
    trainer.dynamic_generation_threshold=50 \
    trainer.dynamic_generation_frequency=20 \
    trainer.openai_api_key="your-api-key-here"
```
