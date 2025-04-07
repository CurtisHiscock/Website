function scrollToSection(id) {
    const section = document.getElementById(id);
    if (section) {
      // Add scrolling class to trigger CSS effect
      document.documentElement.classList.add('scrolling');
  
      //Scroll into view smoothly
      section.scrollIntoView({ behavior: 'smooth' });
  
      setTimeout(() => {
        document.documentElement.classList.remove('scrolling');
      }, 100); //Adjust timing to match scroll speed
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("copyEmail");
    const copiedMsg = document.getElementById("copiedMsg");
    const email = button.textContent;
  
    button.addEventListener("click", () => {
      navigator.clipboard.writeText(email).then(() => {
        // Fade in
        copiedMsg.classList.add("show");
  
        // Fade out after 2 seconds
        setTimeout(() => {
          copiedMsg.classList.remove("show");
        }, 2000);
      });
    });
  });