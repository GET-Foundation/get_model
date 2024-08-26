To create a new model class for use in training, there are several steps involved. Let's break them down while noting inheritance and common structures:

1. Create a Config Class:
   - Define a new dataclass that inherits from BaseConfig
   - Include all necessary parameters for your model
   - Use the @dataclass decorator
   - Example:
     ```python
     @dataclass
     class NewModelConfig(BaseGETModelConfig):
         specific_param: int = 10
         another_param: str = "default"
     ```

2. Create Specific Module Classes (if needed):
   - Inherit from BaseModule
   - Define the __init__ method using the corresponding config
   - Implement the forward method
   - Implement the generate_dummy_data method
   - Example:
     ```python
     class SpecificModule(BaseModule):
         def __init__(self, cfg: SpecificModuleConfig):
             super().__init__(cfg)
             # Initialize layers

         def forward(self, x):
             # Implement forward pass

         def generate_dummy_data(self, batch_size=1):
             # Generate dummy data for testing
     ```

3. Create the Main Model Class:
   - Inherit from BaseGETModel
   - Define the __init__ method using the new config class
   - Initialize all necessary modules
   - Implement the get_input method to prepare input data
   - Implement the forward method
   - Implement the before_loss method to prepare output for loss calculation
   - Implement the generate_dummy_data method
   - Example:
     ```python
     class NewModel(BaseGETModel):
         def __init__(self, cfg: NewModelConfig):
             super().__init__(cfg)
             self.specific_module = SpecificModule(cfg.specific_module)
             # Initialize other modules

         def get_input(self, batch):
             # Prepare input data

         def forward(self, *args, **kwargs):
             # Implement forward pass

         def before_loss(self, output, batch):
             # Prepare output for loss calculation

         def generate_dummy_data(self):
             # Generate dummy data for testing
     ```

4. Create a YAML Config File:
   - Define the model configuration in a YAML file
   - Include all necessary parameters
   - Specify the target class
   - Example:
     ```yaml
     model:
       _target_: path.to.NewModel
       cfg:
         specific_param: 20
         another_param: "custom"
         specific_module:
           param1: value1
           param2: value2
     ```

Common Structures and Inheritance:
- All config classes inherit from BaseConfig
- All module and model classes inherit from BaseModule
- The main model class inherits from BaseGETModel
- Common methods across models: get_input, forward, before_loss, generate_dummy_data
- Use of dataclasses for configuration
- Consistent naming conventions (e.g., Config suffix for config classes)
- Use of type hinting throughout the code

By following these steps and maintaining consistency with the existing structure, you can create a new model class that integrates seamlessly with the training pipeline. The inheritance structure allows for code reuse and maintains a consistent interface across different models and modules.