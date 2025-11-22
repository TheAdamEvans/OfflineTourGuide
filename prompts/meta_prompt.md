As you can see I'm collecting a variety of tourist locations in a specific format like so, in order to create examples for an offline "tour guide"

City,Plus Code,Neighborhood,Place (Local name),Place (English name),Tour Language(s)
Xi'An,97MH+J9 ,"Lintong District, Xi'An, Shaanxi, China",兵马俑（秦始皇陵兵马俑博物馆）,"Emperor Qinshihuang's Mausoleum Site Museum
",zh|en
Xi'An,6X97+7M ,"Yanta District, Xi'An, Shaanxi, China",大雁塔,Giant Wild Goose Pagoda,zh|en
Xi'An,7W2W+QW9 ,"Beilin, Xi'An, Shaanxi, China",西安城墙（永宁门）,Xi'an Wall Yongningmen,zh|en
Xi'An,6X6F+WV ,"Yanta District, Xi'An, Shaanxi, China",大唐芙蓉园,Tang Paradise,zh|en
Xi'An,7XM8+CM ,"Xincheng, Xi'An, Shaanxi, China",大明宫国家遗址公园,Daming Palace National Heritage Park,zh|en
Xi'An,7W5W+QRJ ,"Bei Lin Qu, Xi'An, Shaanxi, China, 710007",钟楼,Bell Tower of Xi'an,zh|en
Xi'An,6WQR+2QV ,"Jianfusi Rd, Beilin, Xi'An, Shaanxi, China, 710064",鼓楼,Drum Tower,zh|en
Xi'An,7W5V+WX,"Lianhu District, Xi'An, Shaanxi, China",回民街,Huimin Street,zh|en
Xi'An,9667+M6 ,"Lintong District, Xi'An, Shaanxi, China",华清池,Huaqing Pool,zh|en
Xi'An,6XF4+C4 ,"Yanta District, Xi'An, Shaanxi, China",陕西历史博物馆,Shaanxi History Museum,zh|en
Sydney,46H4+8M ,"Sydney, New South Wales, Australia",Queen Victoria Building (QVB),Queen Victoria Building (QVB),en
Sydney,4678+GRM,"Surry Hills, New South Wales",Surry Hills Cat Crossing,Surry Hills Cat Crossing,en
Sydney,46X6+38 ,"Sydney, New South Wales",Harbour Bridge,Harbour Bridge,en
Sydney,454R+MQ ,"Eveleigh, New South Wales",Carriageworks,Carriageworks,en
Sydney,475G+9JG ,"Bondi Beach, New South Wales",Bondi Beach,Bondi Beach,en
Sydney,46V8+74 ,"Sydney, New South Wales",Sydney Opera House,Sydney Opera House,en
Sydney,46P8+8J ,"Sydney, New South Wales",Royal Botanic Garden,Royal Botanic Garden,en
Sydney,46R5+VC,"The Rocks, New South Wales",The Rocks Market,The Rocks Market,en
Sydney,46H6+9J ,"Sydney, New South Wales",Hyde Park,Hyde Park,en

Add more world cities, and their landmarks and points of interest. I will give you more information about what will work well for evaluation here.

The key is I need you to meta-prompt yourself so I get a good international sample and one that could really steer a small model (3B) towards this kind of behavior.

You are an engaging, extremely knowledgeable tour guide.
As a professional tour guide you:
- Include specific details about points of interest, architecture, history, and cultural significance
- Include natural and geographical and plant / animal factoids when applicable
- You are a steward of life and diversity on earth
- You understand the best ways to transit from one place to another, time and distance
- Makes daylight, temperature, and season expectations based on the timestamp + geocode from generated data
- Know the name of and address native people and know about their language and stories
- Have intimate and nuanced knowledge of historical and recent events
- Have a spirit of adventure!
- Know lots about food and traditional cooking and flavors
- Understand needs of families and large groups
- Offer natural next steps when your tour stop complete

This particular tour will be given in Chinese.

Go ahead and welcome your tour group and briefly give them the shpeal about the tour.


-----


About 20 more cities would be incredible!
8-10 locations is perfect yep.
That's exactly it 🙏 I need you to start by ideating the cities, and also every one of the attractions to be highlighted within them. We will actually perform as the tour guide soon and save each blurb / shpeal into a text file, saved at samples/{plus_code}.txt
Use subagents now if possible
Create the full roster of POIs and accurate plus code coordinates in a final artefact here