from dataclasses import dataclass
from llm_client import *
from prompts import *
from process import *
import pickle
from xgboost import XGBClassifier

@dataclass
class Decision:
    action: str
    say: str
    args: Dict[str, Any]
def _safe_json(raw):
    try:
        s = (raw or "").strip()
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j != -1 and j > i:
            import json
            return json.loads(s[i:j+1])
    except Exception:
        pass
    return None
class Brain:
    def __init__(self, client):
        self.client = client
        self.states_list = ["REFERENCE","QUERY" ,"TRAIN_OR_PREDICT", "SPLITTING","TRAINING", "PREDICTING"]
    
    def _REFERENCE(self, csv_path):
        csv_sample_text = extract_csv_sample(csv_path)
        prompt = prompt_csv_validation(csv_sample_text)
        answer = self.client.chat(prompt)
        data = _safe_json(answer)
        if data is None:
            action = "REFERENCE"
            say ="Unable to determine, please re-upload a file!"
            x, y = None, None
        else:
            answer = str(data.get("output"))
            if answer == "Yes":
                x, y = data_process(csv_path)
                action = "QUERY"
                say = "The Reference dataset has been handled over! Please provide the query dataset!"
            else:
                action = "REFERENCE"
                say ="The input file seems to be wrong, please re-upload!"
                x, y = None, None
        return Decision(action=action, say=say, args={"x":x, "y":y})
    
    def _QUERY(self, csv_path):
        csv_sample_text = extract_csv_sample(csv_path)
        prompt = prompt_csv_validation(csv_sample_text)
        answer = self.client.chat(prompt)
        data = _safe_json(answer)
        if data is None:
            action = "QUERY"
            say ="Unable to determine, please re-upload a file!"
            x_test, y_test = None, None
        else:
            answer = str(data.get("output"))
            if answer == "Yes":
                x_test, y_test = data_process(csv_path)
                action = "TRAIN_OR_PREDICT"
                say = "The Reference dataset has been handled over! Anotate the cell type by trainning model or directly predict?"
            else:
                action = "QUERY"
                say ="The input file seems to be wrong, please re-upload!"
                x_test, y_test = None, None
        return Decision(action=action, say=say, args={"x_test":x_test, "y_test":y_test})
    def _TRAIN_OR_PREDICT(self, msg):
        prompt = prompt_train_or_predict(msg)
        answer = self.client.chat(prompt)
        data = _safe_json(answer)
        if data is None:
            action = "TRAIN_OR_PREDICT"
            say = "Unable to determine, please re-enter!"
        else:
            decision_map = {
            "Train": ("SPLITTING", "You chose to train, please input the ratio of valid_size!"),
            "Uncertain": ("TRAIN_OR_PREDICT", "Uncertain, please re-enter!"),
            "Predict": ("PREDICTING", "You chose to predict. Begin to predict now?")}
            answer = str(data.get("output"))
            action, say = decision_map.get(answer)
        return Decision(action=action, say=say, args={})
    
    def _SPLITTING(self, msg, x, y):
            prompt = prompt_splitting(msg)
            answer = self.client.chat(prompt)
            data = _safe_json(answer)
            if data is None:
                action = "SPLITTING"
                say ="Unable to determine, please re-input the ratio of valid_size!"
                train_mask, valid_mask = None, None
            else:
                valid_size = float(data.get("valid_size"))
                if valid_size != -1.0:
                    train_mask, valid_mask = split_dataset(x, y, valid_size)
                    action = "TRAINING"
                    say = f"The valid_size is {valid_size}! Begin training the model? Yes or No?"
                else:
                    action = "SPLITTING"
                    say ="The input seems to be wrong, please re-input the ratio of valid_size!"
                    train_mask, valid_mask = None, None
            return Decision(action=action, say=say, args={"train_mask":train_mask, "valid_mask":valid_mask}) 
    
    def _TRAINING(self, msg, x, x_test, y, train_mask, valid_mask):
        prompt = prompt_train_or_not(msg)
        answer = self.client.chat(prompt)
        data = _safe_json(answer)
        if data is None:
            action = "TRAINING"
            say ="Unable to determine, please re-enter!"
        else:
            answer = str(data.get("output"))
            if answer == "No" or answer == "Uncertain":
                action = "TRAINING"
                say ="OK, Wait a moment!"
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                tranning_by_scABiGNN(x, x_test, y, train_mask, valid_mask, device)
                tranning_by_celltypist(x, y, train_mask)
                tranning_by_scVI(x, y, train_mask)
                tranning_by_SVM(x, y, train_mask)

                action = "PREDICTING"
                say ="Trainning model successfully! Begin predicting?"                
        return Decision(action=action, say=say, args={})
    
    def _PREDICTING(self, msg, x, train_x, y, train_y):
        prompt = prompt_predict_or_not(msg)
        answer = self.client.chat(prompt)
        data = _safe_json(answer)
        if data is None:
            action = "PREDICTING"
            say ="Unable to determine, please re-enter!"
        else:
            answer = str(data.get("output"))
            if answer == "No" or answer == "Uncertain":
                action = "PREDICTING"
                say = "OK, Wait a moment!"
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                assesser = XGBClassifier(
                    objective="multi:softmax",
                    num_class=len(np.unique(train_y)),
                    eval_metric="mlogloss",
                    max_depth=3,
                    reg_alpha=0.8,             
                    learning_rate=0.1,          
                    n_estimators=200,
                    subsample=0.9, 
                    colsample_bytree=1,
                    scale_pos_weight=1,
                    seed=0
                )
                assesser = train_assesser(assesser, train_x, train_y, device, x, y)
    
                pred_scABiGNN, metrics_scABiGNN = predicting_by_scABiGNN(x, y, train_x, device,0)
                pred_celltypist, metrics_celltypist = predicting_by_celltypist(x, y)
                pred_scVI, metrics_scVI = predicting_by_scVI(x, y)
                pred_SVM, metrics_SVM = predicting_by_SVM(x, y, train_x)

                X_meta = np.column_stack([pred_scABiGNN,pred_celltypist,pred_scVI.astype(int),pred_SVM])
                pred_AnnoAgent = assesser.predict(X_meta)
                
                with open("pred_AnnoAgent.pkl", "wb") as f:
                    pickle.dump(pred_AnnoAgent, f)   

                metrics_AnnoAgent = calculate_metrics(y, np.array(pred_AnnoAgent).astype(int))
                total_metrics = {
                    "scABiGNN":metrics_scABiGNN,
                    "celltypist":metrics_celltypist,
                    "scVI":metrics_scVI,
                    "SVM":metrics_SVM,
                    "AnnoAgent":metrics_AnnoAgent
                }
                
                action = "REFERENCE"
                say =f"Predicting successfully! The metrics of is {total_metrics}!\n"\
                "Next, please provide the reference dataset!"
        return Decision(action=action, say=say, args={})
    
    def decide(self, msg, arg_paras):
        mode = arg_paras.get("mode")
        if mode == "REFERENCE":
            decision = self._REFERENCE(msg)
            arg_paras["x"] = decision.args.get("x")
            arg_paras["y"] = decision.args.get("y")
        if mode == "QUERY":
            decision = self._QUERY(msg)
            arg_paras["x_test"] = decision.args.get("x_test")
            arg_paras["y_test"] = decision.args.get("y_test")
        if mode == "TRAIN_OR_PREDICT":
            decision = self._TRAIN_OR_PREDICT(msg)
        if mode == "SPLITTING":
            decision = self._SPLITTING(msg, arg_paras.get("x"), arg_paras.get("y"))
            arg_paras["train_mask"] = decision.args.get("train_mask")
            arg_paras["valid_mask"] = decision.args.get("valid_mask")
        if mode == "TRAINING":
            decision = self._TRAINING(msg, arg_paras.get("x"), arg_paras.get("x_test"), arg_paras.get("y"), arg_paras.get("train_mask"), arg_paras.get("valid_mask"))
            if decision.action == "PREDICTING":
                arg_paras["train_mask"] = None
                arg_paras["valid_mask"] = None
        if mode == "PREDICTING":
            decision = self._PREDICTING(msg, arg_paras.get("x_test"), arg_paras.get("x"), arg_paras.get("y_test"), arg_paras.get("y"))
            if decision.action == "REFERENCE":
                arg_paras["x"] = None
                arg_paras["y"] = None   
                arg_paras["x_test"] = None
                arg_paras["y_test"] = None             
        arg_paras["mode"] = decision.action
        return decision.say
