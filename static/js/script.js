document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.querySelector('.nav-toggle');
    const mainNav = document.getElementById('main-nav');

    if (navToggle && mainNav) {
        navToggle.addEventListener('click', function() {
            mainNav.classList.toggle('nav-open');
        });

        // Opcional: Cerrar el menú si se hace clic fuera de él
        document.addEventListener('click', function(event) {
            const isClickInsideNav = mainNav.contains(event.target);
            const isClickOnToggle = navToggle.contains(event.target);

            if (!isClickInsideNav && !isClickOnToggle && mainNav.classList.contains('nav-open')) {
                mainNav.classList.remove('nav-open');
            }
        });

        // Opcional: Cerrar el menú al hacer clic en un enlace (para móviles)
        const navLinks = mainNav.querySelectorAll('.nav-links li a');
        navLinks.forEach(link => {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 768) { // Solo si estamos en móvil
                    mainNav.classList.remove('nav-open');
                }
            });
        });
    }
});