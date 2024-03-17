let positiveAttribution, negativeAttribution
let positiveMaxNum, negativeMaxNum;
let num_pos_shown = 2
let num_neg_shown = 2
let posTfIdf, negTfIdf
let iframeId

let expanded = false;
let expandedAttributedTextWrapperElementId = "";
let expandedAttributedTextElementId = "";
let histogramHighlightedTextWrapperElementIds = [];

document.addEventListener("attribute", function(event) {
    console.log(event)
    positiveAttribution = event.positiveAttribution
    negativeAttribution = event.negativeAttribution
    positiveMaxNum = event.positiveMaxNum
    negativeMaxNum = event.negativeMaxNum
    posTfIdf = event.posTfIdf
    negTfIdf = event.negTfIdf
    iframeId = event.iframeId

    firstTokenSpacing(d3.select(".generated-text .tokens-container"))

    // positive dropdown
    d3.select(".positive-attribution-num-dropdown-wrapper")
        .append("div")
            .attr("class", "attribution-num-dropdown-title-wrapper positive-attribution-num-dropdown-title-wrapper")
            .on("mouseover", function(event) {  // Mouseover Event Listener
                d3.select(".positive-attribution-num-dropdown-svg").style("fill", "var(--blue5)")
            })
            .on("mouseout", function(event) {  // Mouseover Event Listener
                d3.select(".positive-attribution-num-dropdown-svg").style("fill", "var(--blue3)")
            })
            .on("click", function(event) {  // Click Event Listener
                d3.select(".positive-attribution-num-dropdown-options-wrapper")
                    .style("display", d3.select(".positive-attribution-num-dropdown-options-wrapper").style("display")=="none"?"block":"none")  
                document.getElementsByClassName("positive-attribution-num-dropdown-wrapper")[0].expanded = !document.getElementsByClassName("positive-attribution-num-dropdown-wrapper")[0].expanded
            })
            .append("div")
                .attr("class", "attribution-num-dropdown-text positive-attribution-num-dropdown-text")
                .text(num_pos_shown)
    document.getElementsByClassName("positive-attribution-num-dropdown-wrapper")[0].expanded = false
    d3.select(".positive-attribution-num-dropdown-title-wrapper")
        .append("svg")
            .attr("class", "attribution-num-dropdown-svg positive-attribution-num-dropdown-svg")
            .append("g")
            .append("path")
            .attr("d", "M 0 0 L 10 0 L 5 10 Z")
    d3.select(".positive-attribution-num-dropdown-wrapper")
        .append("div")
            .attr("class", "attribution-num-dropdown-options-wrapper positive-attribution-num-dropdown-options-wrapper")
            .selectAll("p")
            .data([...Array(positiveMaxNum+1).keys()])
            .enter()
            .append("p")
                .attr("class", "attribution-num-dropdown-option positive-attribution-num-dropdown-option")
                .text(d=>d)
                .on("mouseover", function(event) {  // Mouseover Event Listener
                    d3.select(this).style("background-color", "#e0e0e0")
                })
                .on("mouseout", function(event) {  // Mouseover Event Listener
                    d3.select(this).style("background-color", "#f0f0f0")
                })
                .on("click", function(event) {  // Click Event Listener
                    num_pos_shown = +d3.select(this).text()
                    d3.select(".positive-attribution-num-dropdown-text").text(num_pos_shown)
                    displayAttribution(positiveAttribution, positive=true)
                    summarizeAttributedData(posTfIdf, true)
                    d3.select(".positive-attribution-num-dropdown-options-wrapper").style("display", "none")
                })

    // negative dropdown
    d3.select(".negative-attribution-num-dropdown-wrapper")
        .append("div")
            .attr("class", "attribution-num-dropdown-title-wrapper negative-attribution-num-dropdown-title-wrapper")
            .on("mouseover", function(event) {  // Mouseover Event Listener
                d3.select(".negative-attribution-num-dropdown-svg").style("fill", "var(--blue5)")
            })
            .on("mouseout", function(event) {  // Mouseover Event Listener
                d3.select(".negative-attribution-num-dropdown-svg").style("fill", "var(--blue3)")
            })
            .on("click", function(event) {  // Click Event Listener
                d3.select(".negative-attribution-num-dropdown-options-wrapper")
                    .style("display", d3.select(".negative-attribution-num-dropdown-options-wrapper").style("display")=="none"?"block":"none")  
                document.getElementsByClassName("negative-attribution-num-dropdown-wrapper")[0].expanded = !document.getElementsByClassName("negative-attribution-num-dropdown-wrapper")[0].expanded
            })
            .append("div")
                .attr("class", "attribution-num-dropdown-text negative-attribution-num-dropdown-text")
                .text(num_neg_shown)
    document.getElementsByClassName("negative-attribution-num-dropdown-wrapper")[0].expanded = false
    d3.select(".negative-attribution-num-dropdown-title-wrapper")
        .append("svg")
            .attr("class", "attribution-num-dropdown-svg negative-attribution-num-dropdown-svg")
            .append("g")
            .append("path")
            .attr("d", "M 0 0 L 10 0 L 5 10 Z")
    d3.select(".negative-attribution-num-dropdown-wrapper")
        .append("div")
            .attr("class", "attribution-num-dropdown-options-wrapper negative-attribution-num-dropdown-options-wrapper")
            .selectAll("p")
            .data([...Array(negativeMaxNum+1).keys()])
            .enter()
            .append("p")
                .attr("class", "attribution-num-dropdown-option negative-attribution-num-dropdown-option")
                .text(d=>d)
                .on("mouseover", function(event) {  // Mouseover Event Listener
                    d3.select(this).style("background-color", "#e0e0e0")
                })
                .on("mouseout", function(event) {  // Mouseover Event Listener
                    d3.select(this).style("background-color", "#f0f0f0")
                })
                .on("click", function(event) {  // Click Event Listener
                    num_neg_shown = +d3.select(this).text()
                    d3.select(".negative-attribution-num-dropdown-text").text(num_neg_shown)
                    displayAttribution(negativeAttribution, positive=false)
                    summarizeAttributedData(negTfIdf, false)
                    d3.select(".negative-attribution-num-dropdown-options-wrapper").style("display", "none")
                })

    displayAttribution(positiveAttribution, positive=true)
    displayAttribution(negativeAttribution, positive=false)

    drawHistogram(event.scoreHistogramCounts)
    resetAttributedDataSummary(posTfIdf, negTfIdf)
})

document.scrollingElement.style.scrollbarGutter = "stable"
document.scrollingElement.style.scrollbarWidth = "thin"

document.addEventListener("mouseup", function(event) {
    // if click is in dropdown box, do nothing
    let posDropdownWrapper = document.getElementsByClassName("positive-attribution-num-dropdown-wrapper")[0]
    let negDropdownWrapper = document.getElementsByClassName("negative-attribution-num-dropdown-wrapper")[0]
    let posDropdownBox = posDropdownWrapper.getBoundingClientRect()
    let negDropdownBox = negDropdownWrapper.getBoundingClientRect()
    let posDropdownBoxLeft = posDropdownBox.x, posDropdownBoxRight = posDropdownBox.x + posDropdownBox.width, posDropdownBoxTop = posDropdownBox.y, posDropdownBoxBottom = posDropdownBox.y + posDropdownBox.height
    let negDropdownBoxLeft = negDropdownBox.x, negDropdownBoxRight = negDropdownBox.x + negDropdownBox.width, negDropdownBoxTop = negDropdownBox.y, negDropdownBoxBottom = negDropdownBox.y + negDropdownBox.height

    if (event.clientX > posDropdownBoxLeft && event.clientX < posDropdownBoxRight && event.clientY > posDropdownBoxTop && event.clientY < posDropdownBoxBottom) return;
    if (event.clientX > negDropdownBoxLeft && event.clientX < negDropdownBoxRight && event.clientY > negDropdownBoxTop && event.clientY < negDropdownBoxBottom) return;

    // if dropdown is expanded, collapse it
    if (posDropdownWrapper.expanded) {
        d3.select(".positive-attribution-num-dropdown-options-wrapper").style("display", "none")
        posDropdownWrapper.expanded = false
    }  
    if (negDropdownWrapper.expanded) {
        d3.select(".negative-attribution-num-dropdown-options-wrapper").style("display", "none")
        negDropdownWrapper.expanded = false
    }  

    // if any attribution box is expanded and click is outside the box, collapse it
    if (expanded) {
        let expandedBox = document.getElementById(expandedAttributedTextWrapperElementId).getBoundingClientRect()
        let expandedBoxLeft = expandedBox.x, expandedBoxRight = expandedBox.x + expandedBox.width, expandedBoxTop = expandedBox.y, expandedBoxBottom = expandedBox.y + expandedBox.height
        if (event.clientX > expandedBoxLeft && event.clientX < expandedBoxRight && event.clientY > expandedBoxTop && event.clientY < expandedBoxBottom) {}
        else {
            collapseExpandedAttributedText(expandedAttributedTextWrapperElementId, expandedAttributedTextElementId);
            if (!event.target.classList.contains("attributed-text-wrapper")) {
                expanded = false;
                expandedAttributedTextWrapperElementId = "";
                expandedAttributedTextElementId = "";
            }
        }
    }
})

function firstTokenSpacing (tokensContainer) {
    tokensContainer.selectAll(".token")
        .each(function() {
            let offsetLeft = document.getElementById(this.id).offsetLeft

            if (offsetLeft < 1) this.classList.add("first-in-line-token")
            else this.classList.remove("first-in-line-token")
        })
}

function displayAttribution (attribution, positive=true) {
    let type = positive?"positive":"negative"
    let num_shown = positive?num_pos_shown:num_neg_shown
    d3.select(`.${type}-attribution`).selectAll(".attributed-text-wrapper").remove()

    for (let idx=0; idx<num_shown; idx++) {
        let i = positive?idx:num_shown-idx-1;
        if (attribution.length <= i) break;
        d3.select(`.${type}-attribution`)
            .append("div")
                .attr("class", `attributed-text-wrapper ${type}-attributed-text-wrapper`)
                .attr("id", `${type}-attributed-text-wrapper-${i}`)
        
        let attributedTextWrapperElement = document.getElementById(`${type}-attributed-text-wrapper-${i}`)
        attributedTextWrapperElement.attributionData = attribution[i]
        attributedTextWrapperElement.positive = positive

        d3.select(`#${type}-attributed-text-wrapper-${i}`)
            .append("div")
                .attr("id", `${type}-attributed-text-info-wrapper-${i}`)
                .attr("class", `attributed-text-info-wrapper ${type}-attributed-text-info-wrapper`)
        d3.select(`#${type}-attributed-text-wrapper-${i}`)
            .append("div")
                .attr("class", `attributed-text ${type}-attributed-text`)
                .attr("id", `${type}-attributed-text-${i}`)
        let attributedTextElement = document.getElementById(`${type}-attributed-text-${i}`)
        setCollapsedText(attribution[i], attributedTextElement)
                

        // Click Event Listener
        d3.select(`.${type}-attribution`).selectAll(".attributed-text-wrapper").on("click", function(event) {
            // tokens-container: for the expanded, attributed-text: for the collapsed
            let clickedWrapperId = event.target.id
            let clickedWrapperElement = document.getElementById(clickedWrapperId)  // wrapper
            if (expanded && clickedWrapperElement.expanded) return;

            let clickedAttributedTextElementId = d3.select(`#${clickedWrapperId}`).select(".attributed-text").attr("id")
            let clickedAttributedTextElement = document.getElementById(clickedAttributedTextElementId)

            let data = clickedWrapperElement.attributionData
            d3.select(`#${clickedWrapperId}`).select(".attributed-text").remove()

            // Expand
            d3.select(`#${clickedWrapperId}`)
                .append("div")
                    .attr("class", "expanded-contents-wrapper")
                    .attr("id", "expanded-contents-wrapper")
                    .style("opacity", 0)
                    .transition()
                    .duration(1000)
                    .style("opacity", 1)
            if (("input_text" in data) && ("output_text" in data)) {
                d3.select(`#${clickedWrapperId}`)
                    .select("#expanded-contents-wrapper")
                    .append("div")
                    .attr("class", "full-text-wrapper")
                d3.select(".full-text-wrapper")
                    .append("div")
                    .attr("class", "input-text-wrapper")
                d3.select(".input-text-wrapper")
                    .append("div")
                    .attr("class", "input-text-title full-text-title")
                    .text("Input")
                d3.select(".input-text-wrapper")
                    .append("div")
                    .attr("class", "input-text-content")
                    .text(data["input_text"])

                d3.select(".full-text-wrapper")
                    .append("div")
                    .attr("class", "output-text-wrapper")
                d3.select(".output-text-wrapper")
                    .append("div")
                    .attr("class", "output-text-title full-text-title")
                    .text("Output")
                d3.select(".output-text-wrapper")
                    .append("div")
                    .attr("class", "output-text-content")
                    .text(data["output_text"])
            }
            else if ("text" in data) {
                d3.select(`#${clickedWrapperId}`)
                    .select("#expanded-contents-wrapper")
                    .append("div")
                    .attr("class", "full-text-wrapper")
                    .html(data["text_html_code"])
                let tokensContainerId = d3.select(`#${clickedWrapperId}`).select(".tokens-container").attr("id")
                firstTokenSpacing(d3.select(`#${tokensContainerId}`))
            }

            for (let key in data) {
                if (key=="text_html_code"||key=="text"||key=="score_histogram_bin"||key=="tokens_container_id"||key=="title"||key=="source") continue
                if (key=="score") continue;
                if (key=="data_index") continue;
                if (key=="prompt_ids" || key =="output_ids" || key=="prompt_text" || key=="output_text" || key=="input_text") continue;

                d3.select(`#${clickedWrapperId}`)
                    .select("#expanded-contents-wrapper")
                        .append("div")
                            .attr("class", `expanded-contents`)
                            .attr("id", `expanded-contents-${key}`)
                d3.select(`#${clickedWrapperId}`)
                    .select(`#expanded-contents-${key}`)
                        .append("span")
                        .attr("class", "expanded-contents-title")
                        .text(key.charAt(0).toUpperCase() + key.slice(1).replace("_", " ")) // + ":")
                d3.select(`#${clickedWrapperId}`)
                    .select(`#expanded-contents-${key}`)
                        .append("span")
                        .attr("class", "expanded-contents-value")
                        .text(data[key])
            }
            if (data["source"]) {
                d3.select(`#${clickedWrapperId}`)
                    .select("#expanded-contents-wrapper")
                        .append("div")
                        .attr("class", "expanded-contents")
                        .attr("id", "expanded-contents-source")
                d3.select(`#${clickedWrapperId}`)
                    .select(`#expanded-contents-source`)
                        .append("span")
                        .attr("class", "expanded-contents-title")
                        .text("Source")
                d3.select(`#${clickedWrapperId}`)
                    .select(`#expanded-contents-source`)
                        .append("a")
                        .attr("class", "expanded-contents-value")
                        .text(data["title"]?data["title"]:"Link")
                        .attr("href", data["source"])
            }
            else if (data["title"]) {
                d3.select(`#${clickedWrapperId}`)
                    .select("#expanded-contents-wrapper")
                        .append("div")
                        .attr("class", "expanded-contents")
                        .attr("id", "expanded-contents-source")
                d3.select(`#${clickedWrapperId}`)
                    .select(`#expanded-contents-title`)
                        .append("span")
                        .attr("class", "expanded-contents-title")
                        .text("Title")
                d3.select(`#${clickedWrapperId}`)
                    .select(`#expanded-contents-title`)
                        .append("span")
                        .attr("class", "expanded-contents-value")
                        .text(data["title"])
            }

            let expandedContentsHeight = document.querySelector(`#${clickedWrapperId} #expanded-contents-wrapper`).getBoundingClientRect().height
            d3.select(`#${clickedWrapperId}`)
                .transition()
                .duration(1000)
                    .style("height", `${expandedContentsHeight+42}px`)
                    .style("pointer", "default")
                    .style("transform", "scale(1)")

            clickedWrapperElement.expanded = true;
            expanded = true;
            expandedAttributedTextWrapperElementId = clickedWrapperId;
            expandedAttributedTextElementId = clickedAttributedTextElementId;

        })

        d3.select(`.${type}-attribution`).selectAll(".attributed-text-wrapper").on("mouseover", function(event) {
            let targetElement = event.target
            let targetId = targetElement.id 
            d3.select(`#${targetId}`).style("cursor", targetElement.expanded?"default":"pointer")

            if (!targetElement.expanded) d3.select(`#${targetId}`).style("transform", "scale(1.01)")
        })

        d3.select(`.${type}-attribution`).selectAll(".attributed-text-wrapper").on("mouseout", function(event) {
            d3.select(`#${event.target.id}`).style("transform", "scale(1)")
        })
    }
}

function collapseExpandedAttributedText (attributedTextWrapperElementId, attributedTextElementId) {
    d3.select(`#${attributedTextWrapperElementId}`)
        .select(".expanded-contents-wrapper")
            .transition()
            .duration(1000)
                .style("opacity", "0")
    setTimeout(function() {
        // rewrite the collapsed text box
        d3.select(`#${attributedTextWrapperElementId}`).html("")
        let attributedTextElementIdSplit = attributedTextElementId.split("-")
        let type = attributedTextElementIdSplit[0]
        let i = attributedTextElementIdSplit[attributedTextElementIdSplit.length-1]
        d3.select(`#${attributedTextWrapperElementId}`)
            .append("div")
                .attr("id", `${type}-attributed-text-info-wrapper-${i}`)
                .attr("class", `attributed-text-info-wrapper ${type}-attributed-text-info-wrapper`)
        d3.select(`#${attributedTextWrapperElementId}`)
            .append("div")
                .attr("class", `attributed-text ${type}-attributed-text`)
                .attr("id", attributedTextElementId)
        let expandedAttributedTextElement = document.getElementById(attributedTextElementId)
        let attribution = document.getElementById(attributedTextWrapperElementId).attributionData
        setCollapsedText(attribution, expandedAttributedTextElement)
        document.getElementById(attributedTextWrapperElementId).expanded = false;
    }, 1000)

    // Collapse the expanded wrapper (size)
    d3.select(`#${attributedTextWrapperElementId}`)
        .transition()
        .duration(1000)
            .style("height", "54px")
}

function setCollapsedText (attribution, element) {
    element.innerText = ""
    let elementId = element.id;
    let elementIdSplit = elementId.split("-")
    let type = elementIdSplit[0]
    let i = elementIdSplit[elementIdSplit.length-1]

    if (("input_text" in attribution) && ("output_text" in attribution)) text = attribution["input_text"]
    else if ("text" in attribution) text = attribution["text"]
    

    d3.select(`#${type}-attributed-text-info-wrapper-${i}`)
        .append("span")
            .attr("class", `attributed-text-info ${type}-attributed-text-info`)
            .html(`#${attribution["data_index"]}`)
    d3.select(`#${type}-attributed-text-info-wrapper-${i}`)
        .append("span")
            .attr("class", `attributed-text-info ${type}-attributed-text-info attributed-text-info-score ${type}-attributed-text-info-score`)
            .html(`Score: ${attribution["score"].toFixed(4)}`)
    
    let word_list = text.split(" ")
    for (word_idx in word_list) {
        let word = word_list[word_idx]
        let original_text = d3.select(`#${element.id}`).text().slice(0,-3)

        if (word_idx==0) d3.select(`#${element.id}`).text(`${word}...`)
        else if (word_idx==word_list.length-1) d3.select(`#${element.id}`).text(`${original_text} ${word}`)
        else d3.select(`#${element.id}`).text(`${original_text} ${word}...`)

        let height = +d3.select(`#${element.id}`).style("height").slice(0,-2)
        let lineheight = +d3.select(`#${element.id}`).style("line-height").slice(0,-2)
        if (height > lineheight*1.5) {
            d3.select(`#${element.id}`).text(`${original_text}...`)
            break
        }
    }
    firstTokenSpacing(d3.select(`#${element.id}`))
}

function drawHistogram(counts) {
    // draw histogram of attribution scores and highlight the bars when mouseover the attribution box
    let margin = {top: 20, right: 25, bottom: 20, left: 15}
    let parentWidth = document.getElementsByClassName("attribution-score-histogram")[0].getBoundingClientRect().width

    let width = parentWidth - margin.left - margin.right
    let height = 120 - margin.top - margin.bottom;

    let lineColor = "#e0e0e0";
    let fillColor = "#d0d0d0";

    let svg = d3.select(".attribution-score-histogram").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    let y = d3.scaleLinear().domain([1,-1]).range([0, height])

    svg.append("g")
        .attr("class", "attribution-score-histogram-y-axis")
        .attr("transform", `translate(0,0)`)
        .append("path")
            .attr("d", `M 0 0 L 0 ${height}`)
            .attr("stroke", lineColor)
            .attr("class", "attribution-score-histogram-y-axis-line")
    d3.select(".attribution-score-histogram-y-axis")
        .append("g")
            .attr("class", "attribution-score-histogram-y-axis-ticks attribution-score-histogram-y-axis-ticks-min")
    d3.select(".attribution-score-histogram-y-axis-ticks-min")
        .append("text")
            .attr("class", "attribution-score-histogram-y-axis-tick-text attribution-score-histogram-y-axis-tick-text-min")
            .text("1")
            .attr("x", -3)
            .attr("y", 4)
    d3.select(".attribution-score-histogram-y-axis")
        .append("g")
            .attr("class", "attribution-score-histogram-y-axis-ticks attribution-score-histogram-y-axis-ticks-max")
    d3.select(".attribution-score-histogram-y-axis-ticks-max")
        .append("text")
            .attr("class", "attribution-score-histogram-y-axis-tick-text attribution-score-histogram-y-axis-tick-text-max")
            .text("-1")
            .attr("x", -3)
            .attr("y", height+4)
    d3.select(".attribution-score-histogram-y-axis")
        .append("g")
            .attr("class", "attribution-score-histogram-y-axis-ticks attribution-score-histogram-y-axis-ticks-zero")
    d3.select(".attribution-score-histogram-y-axis-ticks-zero")
        .append("text")
            .attr("class", "attribution-score-histogram-y-axis-tick-text attribution-score-histogram-y-axis-tick-text-zero")
            .text(0)
            .attr("x", -3)
            .attr("y", height/2+4)
        
    let x = d3.scaleSqrt().range([0,width]).domain([0, Math.max(...counts.map(d=>d[2]))])

    svg.append("g")
        .attr("class", "attribution-score-histogram-bars-wrapper")
        .selectAll("rect")
        .data(counts)
        .enter()
        .append("rect")
            .attr("x", 0)
            .attr("transform", d => `translate(0, ${y(d[1])})`)
            .attr("height", d => y(d[0])-y(d[1]))
            .attr("width", d => x(d[2]))
            .attr("class", (d,i) => `attribution-score-histogram-bar attribution-score-histogram-bar-${i}`)
            .style("fill", fillColor)
            .style("stroke", "white")

    // hovered
    svg.append("text")
        .attr("class", "attribution-score-histogram-hovered-text")
        .style("font-size", "11px")

    svg.append("g")
        .attr("class", "attribution-score-histogram-bar-hovered-wrapper")
        .selectAll("rect")
        .data(counts)
        .enter()
        .append("rect")
            .attr("x", 0)
            .attr("transform", d => `translate(0, ${y(d[1])})`)
            .attr("width", d => width)
            .attr("height", d => y(d[0])-y(d[1]))
            .attr("class", (d,i) => `attribution-score-histogram-bar-hovered attribution-score-histogram-bar-hovered-${i}`)
            .style("fill", "#ffffff00")
            .on("mouseover", function(event) {
                let flag = false
                let bar_index = +this.classList[1].split("-").pop()
                let bin = d3.select(this).data()[0]
                d3.select(`.attribution-score-histogram-bar-${bar_index}`).style("fill", "var(--blue3)")

                svg.select(".attribution-score-histogram-hovered-text")
                    .style("fill", "var(--blue5)")
                    .attr("x", x(bin[2])+3)
                    .attr("y", y(bin[0])-0.5)
                    .text(bin[2])
                
                // loop through all the positive attributed text and highlight in blue as well as the corresponding text box if any
                for (let i=0; i<num_pos_shown; i++) {
                    let attributedTextWrapperElement = document.getElementById(`positive-attributed-text-wrapper-${i}`)
                    let data = attributedTextWrapperElement.attributionData
                    if (data["score_histogram_bin"] == bar_index) {
                        d3.select(`#positive-attributed-text-wrapper-${i}`)
                            .style("border", `3px solid var(--blue6)`)
                            .style("padding", `5.5px 10px`)
                        histogramHighlightedTextWrapperElementIds.push(`positive-attributed-text-wrapper-${i}`)
                        flag = true
                    }
                }
                if (flag) {
                    d3.select(`.attribution-score-histogram-bar-${bar_index}`).style("fill", "var(--blue3)")
                    return
                }
                
                // loop through all the negative attributed text and highlight in red as well as the corresponding the corresponding box if any
                for (let i=0; i<num_neg_shown; i++) {
                    let attributedTextWrapperElement = document.getElementById(`negative-attributed-text-wrapper-${i}`)
                    let data = attributedTextWrapperElement.attributionData
                    if (data["score_histogram_bin"] == bar_index) {
                        d3.select(`#negative-attributed-text-wrapper-${i}`)
                            .style("border", `3px solid var(--blue4)`)
                            .style("padding", `5.5px 10px`)
                        histogramHighlightedTextWrapperElementIds.push(`negative-attributed-text-wrapper-${i}`)
                        flag = true
                    }
                }
                if (flag) {
                    d3.select(`.attribution-score-histogram-bar-${bar_index}`).style("fill", "var(--blue3)")
                    return
                }
            })
            .on("mouseout", function(event) {
                let bar_index = +this.classList[1].split("-").pop()
                svg.select(".attribution-score-histogram-hovered-text").text("")
                d3.select(`.attribution-score-histogram-bar-${bar_index}`).style("fill", fillColor)

                // revert the color of the corresponding text box
                for (let i=0; i<histogramHighlightedTextWrapperElementIds.length; i++) {
                    let id = histogramHighlightedTextWrapperElementIds[i]
                    let positive = id.split("-")[0]=="positive"
                    d3.select(`#${id}`)
                        .style("border", `1px solid var(--blue${positive?4:3})`)
                        .style("padding", `7.5px 12px`)
                }
                histogramHighlightedTextWrapperElementIds = []
            })
}

function resetAttributedDataSummary(posTfIdf, negTfIdf) {
    summarizeAttributedData(posTfIdf, positive=true)
    summarizeAttributedData(negTfIdf, positive=false)
}

function summarizeAttributedData(tfIdf, positive=true) {
    let type = positive?"positive":"negative";
    d3.select(`.${type}-attribution-text-tf-idf`).selectAll("div").remove()
    d3.select(`.${type}-attribution-text-tf-idf-gradient`).remove()
    let numShown = positive?num_pos_shown:num_neg_shown
    let height = document.getElementsByClassName(`${type}-attribution`)[0].getBoundingClientRect().height - 30 - 3
    height = Math.max(height, 100)
    document.getElementsByClassName(`${type}-attribution-text-tf-idf`)[0].style.height = `${height}px`
    if (numShown==0) {
        d3.select(`.${type}-attribution-text-tf-idf`)
            .style("height", "fit-content")
            .append("div")
            .attr("class", `${type}-attribution-text-tf-idf-row attribution-text-tf-idf-row`)
            .text("No data to display")
        return
    }

    data = tfIdf[numShown-1]
    d3.select(`.${type}-attribution-text-tf-idf`)
        .selectAll("div")
        .data(data)
        .enter()
        .append("div")
            .attr("class", `${type}-attribution-text-tf-idf-word attribution-text-tf-idf-word`)
            .text(d => d[0])
            .on("mouseover", function(event) {
                let data = d3.select(this).data()[0]
                d3.select(this)
                    .style("background-color", `var(--blue${positive?2:1})`)
                    .style("border", `2px solid var(--blue${positive?6:4})`)
                    .style("padding", "3px 6px")
                for(let i=0; i<numShown; i++) {
                    if (data[2].includes(i)) {
                        d3.select(`#${type}-attributed-text-wrapper-${i}`)
                            .style("border", `3px solid var(--blue${positive?6:4})`)
                            .style("padding", `5.5px 10px`)
                    }
                }
            })
            .on("mouseout", function(event) {
                d3.select(this)
                    .style("background-color", `var(--blue${positive?1:0})`)
                    .style("border", `1px solid var(--blue${positive?3:2})`)
                    .style("padding", "4px 7px")
                d3.selectAll(`.${type}-attributed-text-wrapper`)
                    .style("border", `1px solid var(--blue${positive?4:3})`)
                    .style("padding", `7.5px 12px`)
            })
    d3.select(`.${type}-attribution-text-tf-idf`)
    .append("div")
    .attr("class", `${type}-attribution-text-tf-idf-dummy-word attribution-text-tf-idf-dummy-word`)
    .style("height", "20px")

    d3.select(`.${type}-attribution-text-tf-idf`)
        .on("mouseover", function(event) {
            d3.select(this).style("scrollbar-width", "thin")
        })
        .on("mouseout", function(event) {
            d3.select(this).style("scrollbar-width", "none")
        })

    d3.select(`.${type}-attribution-text-tf-idf-wrapper`)
        .append("div")
        .attr("class", `${type}-attribution-text-tf-idf-gradient`)
        .style("width", "100%")
        .style("height", "20px")
        .style("position", "absolute")
        .style("bottom", "0")
        .style("background", `linear-gradient(to bottom, #ffffff00, #ffffffff)`)
        .style("pointer-events", "none")
}

function adjustSidebarHeight() {
    let sidebarHeight = document.getElementsByClassName("sidebar")[0].getBoundingClientRect().height
    let attributionHeight = document.getElementsByClassName("attribution-result-wrapper")[0].getBoundingClientRect().height
    let height = Math.max(sidebarHeight, attributionHeight)
    document.getElementsByClassName("sidebar")[0].style.height = `${height}px`
}
