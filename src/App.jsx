import Nav from './components/Nav.jsx'
import Hero from './components/Hero.jsx'
import AcademicBackground from './components/AcademicBackground.jsx'
import Publications from './components/Publications.jsx'
import TeachingOutcomes from './components/TeachingOutcomes.jsx'
import Curriculum from './components/Curriculum.jsx'
import Contact from './components/Contact.jsx'
import Footer from './components/Footer.jsx'

export default function App() {
  return (
    <div className="font-sans">
      <Nav />
      <main>
        <Hero />
        <AcademicBackground />
        <Publications />
        <TeachingOutcomes />
        <Curriculum />
        <Contact />
      </main>
      <Footer />
    </div>
  )
}
