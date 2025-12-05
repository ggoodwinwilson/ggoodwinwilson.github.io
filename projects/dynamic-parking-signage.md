---
title: Dynamic Parking Signage Capstone Project
date: July 2020
---
Introduction
Working with UBC Parking and Access Services, our capstone team engineered a dynamic, wireless parking signage system to automate parking restrictions. The goal was to replace static signage with a programmable e-ink display that allows administrators to remotely manage parking rules with a web dashboard. The system integrates four key subsystems: a Power Architecture that manages solar harvesting and battery backup; a Communication Module for LTE web connectivity; an E-ink Display for energy-efficient imaging; and an Online Management portal. My primary contribution was focused on the electronics and enclosure design, while my teammates focused on firmware, the back-end webserver, user interface, and documentation. 

put image here side by side: 
assets/dynamic-parking-images/front-display.webp
assets/dynamic-parking-images/exploded-view.webp
Subtext: Top-level design of the dynamic parking sign, including aluminum sign and electronics integration

Hardware Design
This section details the engineering decisions behind the parking sign’s hardware. The design prioritizes energy efficiency to meet the strict requirement of solar self-sufficiency while maintaining a compact form factor.

put image here:
assets/dynamic-parking-images/full-system-architecture.webp
Subtext: Full system electrical architecture block diagram

Functional Description
The hardware architecture orchestrates power, processing, and communication. As shown in the system block diagram, the electronics receive power simultaneously from a solar panel and a battery pack. A dedicated regulation circuit steps down these inputs to a steady voltage for the logic rails. The core of the system is an MCU running C firmware, which manages:
- Wireless Communication: An off-the-shelf cellular module receiving commands from the web app.
- Visual Output: An e-ink display acting as the "dynamic" face of the sign.
- Sensor Fusion: Continuous monitoring of motion, temperature, and ambient light to optimize power states.

Power Architecture

The device is required to be purely solar-powered, capable of running for two weeks without sunlight. Therefore, the power architecture is a hybrid system designed to extract maximum efficiency from limited sunlight while relying on a robust battery backup during deficits. To maintain a sleek profile a compact Renogy 10W panel was used. The smaller panel generated less power but it should still be enough to never need to charge the battery. Using Canadian daily mean insolation data, we calculated that even in December (Vancouver’s worst-case month), this panel produces ~262Wh over two weeks. This exceeds our worst-case power budget, ensuring viability even with de-rating factors applied.

put image here:
assets/dynamic-parking-images/power-architecture.webp
Subtext: Power flow architecture block diagram

The main power path accepts input from either the solar panel or an external 19-24V barrel jack (for wall power/debugging). A diode-OR controller was used to choose the right input. Unlike a traditional diode OR configuration, this IC controls the gate of a MOSFET, significantly reducing voltage drop and thus power loss.

Following the input stage, a buck converter provides a 5V rail for the system, with a linear regulator providing a more precise 3.3V line where necessary. We also decided to support a USB input to allow for direct to PC UART communication and benchtop powering without the need for the battery pack.

Max Power Point Tracking (MPPT)
With only 10W of available solar capacity, every milliwatt counts. When using a solar panel, it's important to operate at the max power point on the solar panel to extract as much energy as possible. Therefore, a circuit to dynamically adjust the current draw to match the solar panel's optimal voltage point is necessary. The LT3652 was used, a power tracking 2A battery
charger for solar powered input. While chips like the SM72441 offer more versatility, they require complex external bridging. The LT3652 provided a robust, turnkey solution for our <2A current requirements, offering ~80-88% efficiency and simplifying the PCB layout.

Battery Selection & Configuration 
Because our average power draw is low (~0.94W), energy density was prioritized over power density. Panasonic NCR18650B Lithium-Ion cells meet our total energy requirement. Considering cold weather capacity degredation of li-ion batteries, at -10°C (Vancouver winter low), these cells still retain sufficient capacity to meet our needs.

Cells are arranged in a 2-series configuration (2S) to create a pack voltage (5.0V–8.4V) that sits perfectly within the MPPT’s float voltage output range. Parallel strings were added to achieve the required 54Wh capacity buffer, ensuring the two-week runtime requirement is met with a safety margin.

Battery Management System
Safety is paramount with most lithium-ion chemistry since they can be sensitive to mis-use. Therefore it was necessary to add battery management circuitry to protect the cells from electrical overstress conditions.

To protect the cells from a downstream short-circuit, individual 1.5A fuses are used at the output of each cell. Comparators circuits check for under-voltage/over-temperature, and can cut off power with a high-side n-channel MOSFET. The same MOSFET is utilized for reverse polarity protection (putting the batteries in backwards). To keep the series cell voltages balanced, A TI BQ29209 IC automatically ensures the charge equal between cells to prevent dangerous overcharging of a single cell.

Communication
Our requirement was to change the image displayed by the signage in under 2 minutes. A 4G LTE Quectel BG96 modem can achieve the required bitrate, with CAT M1 LTE at 375kbps. The modem form factor is a standalone module mounted via an expansion board from the STM32 P-L496G-CELL02 pack, ensuring seamless compatibility with our STM32 microcontroller ecosystem, and comes with firmware support. There's also a way to put the modem to sleep in order to reduce quiescent power draw. 

Display
To minimize consumption, a Goodisplay GDEW1248Z95 e-ink screen was used. The primary advantage of e-ink is bistability — it consumes zero power to hold a static image, only drawing current when pixel states are changed. Rather than using a separate, bulky driver board, I integrated the display driving circuitry directly onto the main PCB. This reduced the overall enclosure volume, lowered BOM costs, and simplified the enclosure design. 

put image here:
assets/dynamic-parking-images/assy-no-screen.webp
Subtext: E-ink display integration in the electronics enclosure

The perimeter of the display can be lit with LEDs to see the signage when dark. Since running it consumes ~4.3W, it needs to be toggled on and off at the right time. I implemented a "smart lighting" system using a PIR motion sensor and a photoresistor. The LEDs only activate when ambient light is low and motion is detected (e.g. a car parking). This reduced average consumption to 0.43W, saving 90% of the lighting power budget.

Microcontroller
The STM32L496 (Cortex M4) was used for this design. This was largely driven by code migration strategy: the complex LTE cellular stack was already written for the STM32L4 family. It was significantly easier to port the simpler display drivers to the L4 than to port the cellular stack to the legacy F103 chip. The L4 also offers superior low-power sleep modes, essential to avoid manually charging the battery.

PCBA Integration

put image here:
assets/dynamic-parking-images/board-section-breakdown.webp
Subtext: PCBA circuit integration and placement

All subsystems—power, BMS, MCU, and display drivers—were integrated into a single custom PCB. This eliminates the need for most wiring (which can be expensive and unreliable), reduced the device footprint, and allows for easier manufacturing. Crucially, integrating the display driver saved ~1cm of vertical height, allowing us to stay within the 2-inch thickness constraint.

put images side by side here:
assets/dynamic-parking-images/electronics-top.webp
assets/dynamic-parking-images/electronics-iso.webp
Subtext: Views of the PCBA inside the enclosure 

Enclosure Design
The enclosure had to be waterproof (IP65), compact, and retrofittable to standard aluminum parking signs. I designed a custom enclosure that sandwiches the PCB and e-ink display. It features a window for the screen and mounts directly to the existing aluminum signage cutout. Weatherproofing is achieved through a combination of rubber gaskets, epoxy potting, and silicone sealant. Selective Laser Sintering (SLS) for enclosure fabrication. SLS is non-porous (waterproof) and isotropic, unlike FDM which suffers from micro-cracks between layers, allowing water ingress. Eventually the design should be tooled for injection moulding or some other process, but 3D printing worked ok for the proof of concept design.

Project Result
I was able to validate most functionality on Revision A of the PCBA, and build 3 full units successfully. Overall the device is capable of receiving a command over the air and actuating the display, which verifies that many parts are working properly including the LTE modem, microcontroller, battery circuit, and downstream power components. 

Rev. A of the design does have a few issues to be addressed. The bottom half of the display does not show the correct image - the pixels are distorted. 

put image here
assets/dynamic-parking-images/front-display-bug.webp
Subtext: Display firmware bug, image is upside down and upper left/right sides are not driven correctly

We suspect a firmware issue since the same behavior is present on all 3 units. The solar power MPPT circuit does not work as intended. It is able to produce an output voltage, but the diode overheats and eventually smokes. The root cause is still to be identified. USB communciation does not work yet due to firmware immiturity. We are trying to combine a lot of systems and libraries including low-level display driver, real-time operating system handling LTE communication, and USB interrupts. Some debugging is necessary to get everything integrated properly without bugs.

Overall this is a decent result for a first pass of the design. Bugs can be investigated thoroughly on Rev. A since basic functionality is working. We're excited to see how far teams take this project in the future. 

arrange images here
assets/dynamic-parking-images/electronics-side.webp
assets/dynamic-parking-images/two-units.webp