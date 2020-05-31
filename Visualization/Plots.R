library(readr)
library(highcharter)

NDCG = read_csv('NDCG.csv')
HR = read_csv('HR.csv')

NDCG  %>% 
  hchart(type = "line", hcaes(x = Iteration, y = Value, group = Model)) %>% 
  hc_xAxis(title = list(text = "Iteration"), max = 20, tickInterval = 2, min = 0) %>% 
  hc_yAxis(title = list(text = "NDCG@10")) %>% 
  hc_title(text = "NDCG@10 for NeuMF Expert Recommendation")

HR  %>% 
  hchart(type = "line", hcaes(x = Iteration, y = Value, group = Model)) %>% 
  hc_xAxis(title = list(text = "Iteration"), max = 20, tickInterval = 2, min = 0) %>% 
  hc_yAxis(title = list(text = "HR@10")) %>% 
  hc_title(text = "HR@10 for NeuMF Expert Recommendation")
