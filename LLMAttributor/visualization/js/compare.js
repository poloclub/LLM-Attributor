let gPositiveAttribution, gNegativeAttribution, uPositiveAttribution, uNegativeAttribution;
let gScoreHistogramBins, uScoreHistogramBins;
let gScoreHistogramCounts, uScoreHistogramCounts;
let gPosTfIdf, gNegTfIdf, uPosTfIdf, uNegTfIdf;
let positiveMaxNum, negativeMaxNum;
let num_pos_shown = 3
let num_neg_shown = 0
let expanded = false;
let expandedAttributedTextWrapperElementId = "";
let expandedAttributedTextElementId = "";
let histogramHighlightedTextWrapperElementIds = [];

document.addEventListener("compare", function(event) {
    console.log(event)

    gPositiveAttribution = event.gPositiveAttribution
    gNegativeAttribution = event.gNegativeAttribution
    uPositiveAttribution = event.uPositiveAttribution
    uNegativeAttribution = event.uNegativeAttribution
    gScoreHistogramBins = event.gScoreHistogramBins
    uScoreHistogramBins = event.uScoreHistogramBins
    gScoreHistogramCounts = event.gScoreHistogramCounts
    uScoreHistogramCounts = event.uScoreHistogramCounts
    gPosTfIdf = event.gPosTfIdf
    gNegTfIdf = event.gNegTfIdf
    uPosTfIdf = event.uPosTfIdf
    uNegTfIdf = event.uNegTfIdf
    positiveMaxNum = event.positiveMaxNum
    negativeMaxNum = event.negativeMaxNum
    
    addAttributionNumDropdown(true, true)
    addAttributionNumDropdown(true, false)
    addAttributionNumDropdown(false, true)
    addAttributionNumDropdown(false, false)
})

function addAttributionNumDropdown(generated, positive) {
    let textType = generated ? "generated" : "user-provided"
    let positiveType = positive ? "positive" : "negative"
    let color = generated ? "blue":"red"
    
    d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-wrapper`)
        .append("div")
        .attr("class", `attribution-num-dropdown-title-wrapper ${textType}-${positiveType}-attribution-num-dropdown-title-wrapper`)
        .style("border-bottom", `1px solid var(--${color}7)`)
        .on("mouseover", function(event) {  // Mouseover Event Listener
            d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-svg`).style("opacity", 0.7)
            d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-text`).style("opacity", 0.7)
        })
        .on("mouseout", function(event) {  // Mouseover Event Listener
            d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-svg`).style("opacity", 1)
            d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-text`).style("opacity", 1)
        })
        .on("click", function(event) {  // Click Event Listener
            d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-options-wrapper`)
                .style("display", d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-options-wrapper`).style("display")=="none"?"block":"none")  
            document.getElementsByClassName(`${textType}-${positiveType}-attribution-num-dropdown-wrapper`)[0].expanded = !document.getElementsByClassName(`${textType}-${positiveType}-attribution-num-dropdown-wrapper`)[0].expanded
        })
        .append("div")
            .attr("class", `attribution-num-dropdown-text ${textType}-${positiveType}-attribution-num-dropdown-text`)
            .text(positive ? num_pos_shown : num_neg_shown)

    document.getElementsByClassName(`${textType}-${positiveType}-attribution-num-dropdown-wrapper`)[0].expanded = false

    d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-title-wrapper`)
        .append("svg")
        .attr("class", `attribution-num-dropdown-svg ${textType}-${positiveType}-attribution-num-dropdown-svg`)
        .attr("width", 10)
        .attr("height", 10)
        .append("path")
            .attr("d", "M 0 0 L 10 0 L 5 10 Z")
            .attr("fill", `var(--${color}3)`)
    d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-wrapper`)
        .append("div")
            .attr("class", `attribution-num-dropdown-options-wrapper ${textType}-${positiveType}-attribution-num-dropdown-options-wrapper`)
            .style("display", "none")
            .selectAll("p")
            .data([...Array(positive?positiveMaxNum+1:negativeMaxNum+1).keys()])
            .enter()
            .append("p")
                .attr("class", `attribution-num-dropdown-option ${textType}-${positiveType}-attribution-num-dropdown-option`)
                .style("background-color", `var(--${color}0)`)
                .style("color", `var(--${color}3)`)
                .style("border", `1px solid var(--${color}1)`)
                .text(d => d)
                .on("mouseover", function(event) {  // Mouseover Event Listener
                    d3.select(this).style("background-color", `var(--${color}1)`)
                })
                .on("mouseout", function(event) {  // Mouseover Event Listener
                    d3.select(this).style("background-color", `var(--${color}0)`)
                })
                .on("click", function(event) {  // Click Event Listener
                    if (positive) num_pos_shown = +d3.select(this).text()
                    else num_neg_shown = +d3.select(this).text()
                    d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-text`).text(d3.select(this).text())
                    displayAttribution(generated, positive)
                    d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-options-wrapper`).style("display", "none")
                    document.getElementsByClassName(`${textType}-${positiveType}-attribution-num-dropdown-wrapper`)[0].expanded = false
                })
}

function displayAttribution(generated, positive) {
    let textType = generated ? "generated" : "user-provided"
    let positiveType = positive ? "positive" : "negative"
    let color = generated ? "blue":"red"
    let num_shown = positive ? num_pos_shown : num_neg_shown
    let attribution = positive ? (generated ? gPositiveAttribution : uPositiveAttribution) : (generated ? gNegativeAttribution : uNegativeAttribution)

    d3.select(`.${textType}-${positiveType}-attribution`).selectAll(".attributed-text-wrapper").remove()

    for (let i=0 ; i<num_shown ; i++) {
        if (attribution.length <= i) break;
        d3.select(`.${textType}-${positiveType}-attribution`)
            .append("div")
            .attr("class", `attributed-text-wrapper ${textType}-${positiveType}-attributed-text-wrapper`)
            .attr("id", `${textType}-${positiveType}-attributed-text-wrapper-${i}`)

        let attributedTextWrapperElement = document.getElementById(`${textType}-${positiveType}-attributed-text-wrapper-${i}`)
        attributedTextWrapperElement.attributionData = attribution[i]
        attributedTextWrapperElement.positive = positive 
        attributedTextWrapperElement.generated = generated

        d3.select(`#${textType}-${positiveType}-attributed-text-wrapper-${i}`)
            .append("div")
                .attr("class", `attributed-text-info-wrapper ${textType}-${positiveType}-attributed-text-info-wrapper`)
                .attr("id", `${textType}-${positiveType}-attributed-text-info-wrapper-${i}`)
        d3.select(`#${textType}-${positiveType}-attributed-text-wrapper-${i}`)
            .append("div")
                .attr("class", `attributed-text ${textType}-${positiveType}-attributed-text`)
                .attr("id", `${textType}-${positiveType}-attributed-text-${i}`)
        let attributedTextElement = document.getElementById(`${textType}-${positiveType}-attributed-text-${i}`)
        setCollapsedText() // TODO

        d3.select(`#${textType}-${positiveType}-attribution`)
            .selectAll(".attributed-text-wrapper")
            .on("click", function(event) {  // Click Event Listener
                let clickedWrapperId = event.target.id 
                let clickedWrapperElement = document.getElementById(clickedWrapperId)
                if (expanded && clickedWrapperElement.expanded) return;

                let clickedAttributedTextElementId = d3.select(`#${clickedWrapperId}`).select(`.attributed-text`).attr("id")
                let data = clickedWrapperElement.attributionData
                d3.select(`#${clickedWrapperId}`).html("")

                // Expand: TODO

            })

    }
    

// TODO
}