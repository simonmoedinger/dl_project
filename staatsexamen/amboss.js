var fall = ""
var final = ""

while (true) {
	try {
		
		var fall_btn = document.getElementsByClassName("shared-415937358--full-2146696451--compact-3772180877--flatBordered-2740782582--rounded-4202758384--flexPadding-2398322637")
		if (fall_btn.length > 0) {
			fall_btn = fall_btn[0]
			fall_btn.click()
		} else {
			fall_btn = null
		}
		
		if ((fall_btn != null && document.getElementsByClassName("thumbnailBox--AKvuO")[0].innerHTML !== "" ) || document.getElementsByClassName("mediaContainer--F7hpm").length > 0) {
			document.getElementsByClassName("buttonLabel--uo9Ma")[1].click()
			continue
		}
		if (fall_btn != null) {
			var current_fall = document.getElementsByClassName("caseStudyContent--JQBQ4")[0].innerHTML.replaceAll("<p>","").replaceAll("</p>","").replaceAll('"',"").replaceAll("\n"," ")
			if(current_fall  !== fall) {
				fall = current_fall
				final += "<Fall>" + fall
			}
		} else {
			final += "<Fall>"
		}
		
		
		final += "<Frage>" + document.getElementsByClassName("questionContent--gsDVn")[0].innerText.replaceAll("\n"," ")
		
		for (x of document.getElementsByClassName("css-84ablj e1tnky2o0")) {
			if(x.innerText === "LÖSUNG ZEIGEN") {
				x.parentElement.parentElement.parentElement.click()
				
			}
		}
		var counter = 0
		var correct_answer = ""
		for (let choice in ["a", "b", "c", "d", "e"]) {
			final += "<" + choice + ">" + choice + ") " + document.getElementsByClassName("answerContent--QI92v")[counter].innerText;
			if (document.getElementsByClassName("answerContent--QI92v")[counter].parentElement.parentElement.parentElement.getAttribute("data-e2e-test-id") === "answer-theme-answerOptionCorrect") {
			correct_answer = choice
			}
			counter++
		}
		final += "<Antwort>" + correct_answer;
		if (document.getElementsByClassName("buttonLabel--uo9Ma")[1].innerHTML === "Zur Auswertung") {
			break
		}
		document.getElementsByClassName("buttonLabel--uo9Ma")[1].click()
		await new Promise(r => setTimeout(r, 300));
	} catch (error) {
		await new Promise(r => setTimeout(r, 1000));
	}
	
}
console.log(final)
