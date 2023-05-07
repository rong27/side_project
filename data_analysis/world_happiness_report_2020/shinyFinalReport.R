library(shiny)
library(shinydashboard)

library(readr)
library(ggplot2)
library(dplyr)

library(rgeos)

library(maptools)
library(ggmap)
library(scales)
library(RColorBrewer)


library(rio)
library(magrittr)
library(data.table)
library(leaflet)

ui <- dashboardPage (
    skin = "black",
    dashboardHeader(title = "The Asia Happiest Top5 Country vs. Taiwan ", titleWidth = 600),
    
    #左側邊攔
    #siderMenu是選單
    dashboardSidebar(
        sidebarMenu(id = 'tabs'),
        menuItem("Year 2020", tabName = "Happy", icon = icon("laugh-wink"))
    ),
    
    #主頁面
    dashboardBody(
        fluidRow(
            box(title = "Map - The Happiest Index of Asia Country",  solidHeader = TRUE, leafletOutput("map", height = 780), width = 6),
            box(title = "The Asia Happiest Top5 Country", soliderHeader = TRUE, plotlyOutput("top5", height =350),width =6),
            # box(title = "Map The Happiest Index of Asia Top5 Country", solidHeader = TRUE, plotlyOutput("region", height = 320), width = 7),
            box(title = "The Asia Happiest Top5 Country with 6 variables", solidHeader = TRUE, plotlyOutput("variables", height = 350), width = 6)
        ),
        tags$footer("Source: Kaggle - World Happiness Report up to 2020")
    )
)





server <- function(input, output){
    
    # 2020 Happiest data
    setwd("C:/Users/rong/Desktop/NKUST/R/2020碩一期末報告R")
    happy2020 <- fread("2020WorldHappinessReport.csv")
    happy2020DT <- data.table(happy2020)

    # 把亞洲國家篩出來
    asiaHappy2020DT <- happy2020DT[grepl("Asia", `Regional indicator`)]
    asiaHappy2020Top5 <- asiaHappy2020DT %>% arrange(desc(`Ladder score`)) %>% slice_head(n = 5)
    asiaHappy2020Top5.center <- data.frame(x = c(120, 103, 121, 100, 126),
                                           y = c(23, 1,  14, 13, 37))
    combine <- data.frame(asiaHappy2020Top5, asiaHappy2020Top5.center)
    # head(asiahappy2020DT$`Country name`)
    # tail(asiahappy2020DT$`Country name`)
    
    # Map - The Happiest Index of Asia Country
    
    # Use leafletOutput() to create a UI element, and renderLeaflet() to render the map widget
    # readShapeSpatial exists in library(maptools)
    # shp是shapefile的縮寫,在GIS領域廣為使用多年
    output$map <- renderLeaflet({
        
        # happyIcon <- makeIcon(iconUrl = "C:/Users/rong/Desktop/NKUST/R/2020碩一期末報告R/happy.png")
                                   
                                  
        leaflet(combine)  %>%
            setView(120.9739,23.5, zoom = 4) %>%
            addTiles() %>%
            addProviderTiles('Stamen.Watercolor') %>% 
            addMarkers(lng = ~x, lat = ~y,
                       popup = ~combine$Country.name,
                       # icon = happyIcon,
                       clusterOptions = markerClusterOptions()) 
        })  #map end
    
    
    
    # Figure (第2格跟第4格)
    # top 5 barchart
    output$top5 <- renderPlotly ({
        
        #亞洲地區幸福指數前五名國家
        asiaHappy2020Top5 <- asiaHappy2020DT %>% arrange(desc(`Ladder score`)) %>% slice_head(n = 5)
        
        #Figure
        #亞洲地區幸福指數前五名國家圖
        asiaTop5Fig <- ggplotly(ggplot(asiaHappy2020Top5, aes( x = `Country name`,label = round(`Ladder score`,4), y = `Ladder score`)) +
                                    geom_bar(fill='#6A6AFF', col='#6A6AFF', stat = 'identity') +
                                    geom_text(data= asiaHappy2020Top5, aes( x = `Country name`, y = `Ladder score`), size = 6, color = 'red')) 
                                    # ggtitle("The happiest Asia Country Top5")) ; asiaTop5Fig
        asiaTop5Fig
        
    }) # top5 end
    
    
    # top 5 variables
    output$variables <- renderPlotly ({
        
        #General 
        # asiaTop5VariablesFig <- plot_ly(asiaHappy2020Top5, x=~`Country name`, y=~`Ladder score`, type = 'bar', name = 'Ladder Score') %>%
        #     add_trace(y=~`Logged GDP per capita`, name = 'GDP') %>%
        #     add_trace(y=~`Social support`, name = 'Social support') %>%
        #     add_trace(y=~`Healthy life expectancy`, name = 'Healthy') %>%
        #     add_trace(y=~`Freedom to make life choices`, name = 'Freedom') %>%
        #     add_trace(y=~`Generosity`, name = 'Generosity') %>%
        #     add_trace(y=~`Perceptions of corruption`, name = 'Corruption') %>%
        #     layout(xaxis = list(title = 'Country'), yaxis = list(title = 'Score'), barmode = 'stack'); asiaTop5VariablesFig

        
        #Explained by (原始圖不好判讀, 改用處理過後的值)
        asiaTop5VariablesFig <- plot_ly(asiaHappy2020Top5, x=~`Country name`, y=~`Dystopia + residual`, type = 'bar', name = 'Ladder Score') %>%
            add_trace(y=~`Explained by: Log GDP per capita`, name = 'GDP') %>%
            add_trace(y=~`Explained by: Social support`, name = 'Social support') %>%
            add_trace(y=~`Explained by: Healthy life expectancy`, name = 'Healthy') %>%
            add_trace(y=~`Explained by: Freedom to make life choices`, name = 'Freedom') %>%
            add_trace(y=~`Explained by: Generosity`, name = 'Generosity') %>%
            add_trace(y=~`Explained by: Perceptions of corruption`, name = 'Corruption') %>%
            layout(xaxis = list(title = 'Country'), yaxis = list(title = 'Score'), barmode = 'stack'); asiaTop5VariablesFig
        
    }) # top5 var end
    
}     #server end       
    
shinyApp(ui = ui, server = server)

