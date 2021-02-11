import src.extract_sales as extract_sales
import src.create_learning as create_learning
import src.model_strategy as model_strategy
import src.stacking as stacking


if __name__ == "__main__":
    extract_sales.extract_sales()
    create_learning.create_learning()
    model_strategy.model_strategy()
    stacking.stacking()
