/**
 * Credit Card Approval Screening Engine - Client Side Interaction Script
 */

document.addEventListener("DOMContentLoaded", function () {
    // 1. Sidebar Toggler Control
    const menuToggle = document.getElementById("menu-toggle");
    if (menuToggle) {
        menuToggle.addEventListener("click", function (e) {
            e.preventDefault();
            document.getElementById("wrapper").classList.toggle("toggled");
        });
    }

    // 2. Form Validation & Interactive Logic (only execute on screening form page)
    const screeningForm = document.getElementById("screening-form");
    if (screeningForm) {
        // Form Validation Interceptor
        screeningForm.addEventListener("submit", function (event) {
            if (!screeningForm.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            screeningForm.classList.add("was-validated");
        }, false);

        // Interactive Fields mapping based on Employment status
        const isUnemployedSelect = document.getElementById("is_unemployed");
        const employmentStartContainer = document.getElementById("employment-start-container");
        const employmentStartInput = document.getElementById("employment_start");
        const occupationSelect = document.getElementById("occupation_type");

    function toggleEmploymentFields() {
            const isUnemployed = isUnemployedSelect.value === "1";
            
            if (isUnemployed) {
                // If Unemployed: hide & disable employment start date
                employmentStartContainer.style.display = "none";
                employmentStartInput.required = false;
                employmentStartInput.value = ""; // Clear input value
                
                // Set occupation to Unknown and visually grey it out
                // Keep it enabled so the value is submitted with the form
                occupationSelect.value = "Unknown";
                occupationSelect.style.opacity = "0.6";
                occupationSelect.style.pointerEvents = "none";
            } else {
                // If Employed: show & enable employment start date
                employmentStartContainer.style.display = "block";
                employmentStartInput.required = true;
                
                // Re-enable occupation dropdown select
                occupationSelect.style.opacity = "1";
                occupationSelect.style.pointerEvents = "auto";
            }
        }

        // Initialize state on page load
        if (isUnemployedSelect) {
            toggleEmploymentFields();
            
            // Listen for changes
            isUnemployedSelect.addEventListener("change", toggleEmploymentFields);
        }
    }

    // 3. Dynamic Progress Bar Styling
    const progressBar = document.querySelector(".progress-bar");
    if (progressBar) {
        const val = progressBar.getAttribute("aria-valuenow");
        if (val) {
            progressBar.style.width = val + "%";
        }
    }
});
