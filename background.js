chrome.runtime.onInstalled.addListener(function (details) {
    if (details.reason === 'install') { // Open the options page after install
      chrome.tabs.create({ url: 'welcome.html' });
    }
    else if (details.reason === 'update' && /^(((0|1)\..*)|(2\.(0|1)(\..*)?))$/.test(details.previousVersion)) { // Clear data from versions before 2.1
      
    }
  });
 
/*chrome.tabs.onUpdated.addListener(
  function(tab) {

   
      fetch('http://127.0.0.1:8000/predict/{feature}?features='+url)
            .then(response => response.text())
            .then(data => {
              let dataObj = JSON.parse(data);
              console.log(JSON.stringify(dataObj));
              //senderResponse({data: dataObj, index: message.index});
            })
            .catch(error => console.log("error", error))
            console.log("Done");
        return true;

      
    }
    
  
);
  
/*chrome.runtime.onMessage.addListener(function(message, sender, senderResponse){
  if(message.msg === "image"){
    
   fetch('http://127.0.0.1:8000/predict/{feature}?features='+message.myurl)
          .then(response => response.text())
          .then(data => {
            let dataObj = JSON.parse(data);
            console.log(JSON.stringify(dataObj));
            //senderResponse({data: dataObj, index: message.index});
          })
          .catch(error => console.log("error", error))
          console.log("Done");
      return true;
  }
});

chrome.runtime.onMessage.addListener(function(rq, sender, sendResponse) {
  // setTimeout to simulate any callback (even from storage.sync)
  setTimeout(function() {
      console.log(rq.data);
      sendResponse({status: true});
  }, 1);
  return true;  // uncomment this line to fix error
});
*/