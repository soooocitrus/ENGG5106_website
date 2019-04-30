var express = require('express');
var app = express();
var fs = require("fs");
var cheerio = require("cheerio");
var bodyParser = require('body-parser');
var multer  = require('multer');
var spawn = require("child_process").spawn;
 
app.use('/public', express.static('public'));
app.use(bodyParser.urlencoded({ extended: false }));
app.use(multer({ dest: './store'}).array('image'));
//home page
app.get('/home.html', function (req, res) {
   res.sendFile( __dirname + "/" + "home.html" );
})
//handle post request
app.post('/file_upload', function (req, res) {
	// get the text
	input_text = req.body.input_text;

	// get the image
    fs.rename(req.files[0].path, "./store/" + req.files[0].originalname, function(err) {
        if (err) {
            throw err;
        }
        console.log('save pic done!');
    })

    //console some log
	console.log(input_text); 
	console.log(req.files[0].originalname);

	// send back data
	var pythonProcess = spawn('python',[process.cwd() + "/script.py", input_text, req.files[0].originalname]);
	pythonProcess.stdout.on('data', function(data){
		var sendBackData = data.toString();
		console.log(sendBackData);
		sendBackData = sendBackData.split("answeris")[1].split(" ");
		console.log("python send back " + sendBackData);		
		res.writeHead(200,{'Content-Type':'text/html'})

		fs.readFile('./pic.html','utf-8',function(err,data){
			if(err){
				throw err ;
			}
		    var $ = cheerio.load(data);
		    for(var i = 0; i < 10; i++) {
			    $("img#img" + i).each(function(j, e) {
			        $(e).attr("src", "/public/images/" + sendBackData[i]);
			    });
			}
			res.end($.html());
		})
	});	
})
 
var server = app.listen(8081, function () {
 
  var host = server.address().address
  var port = server.address().port
 
  console.log("access address http://%s:%s", host, port)
 
})