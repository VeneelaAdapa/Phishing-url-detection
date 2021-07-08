
chrome.tabs.query({active: true, lastFocusedWindow: true}, tabs => {
        let url = tabs[0].url;
        let url_no_protocol = url.replace(/^https?\:\/\//i, "");
       $.getJSON('http://127.0.0.1:8000/predict/{feature}?features='+url_no_protocol, function(datajson) {
        var data= document.getElementById('display');
        data.innerHTML = JSON.stringify(datajson, undefined, 4);
       
        });
        });
