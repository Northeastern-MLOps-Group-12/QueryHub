import React, { useEffect } from "react";
import { Container, Button, Row, Col, Card } from "react-bootstrap";
import { useLocation, useNavigate } from "react-router-dom";
import useAuth from "../hooks/useAuth";
import { heroContent, steps, features, ctaContent } from "../data/homeContent";

// Feature Card Component
interface FeatureCardProps {
  icon: string;
  title: string;
  description: string;
}

// Individual Feature Card
const FeatureCard: React.FC<FeatureCardProps> = ({
  icon,
  title,
  description,
}) => (
  <Col md={4} className="mb-4 d-flex">
    <Card className="text-center shadow-sm w-100 border-0">
      <Card.Body className="d-flex flex-column align-items-center">
        <div
          className="bg-primary text-white rounded-circle d-flex align-items-center justify-content-center mb-3"
          style={{ width: "64px", height: "64px", fontSize: "2rem" }}
        >
          <i className={`bi ${icon}`}></i>
        </div>
        <Card.Title as="h3" className="h5 fw-bold">
          {title}
        </Card.Title>
        <Card.Text className="text-muted">{description}</Card.Text>
      </Card.Body>
    </Card>
  </Col>
);

// Home Page Component
export default function HomePage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated } = useAuth();

  // Handle Get Started button click
  const handleGetStarted = () => {
    if (isAuthenticated) {
      navigate("/chatinterface");
    } else {
      navigate("/account/signin");
    }
  };

  // Scroll to section if URL has a hash
  useEffect(() => {
    if (location.hash) {
      const el = document.getElementById(location.hash.slice(1));
      if (el) el.scrollIntoView({ behavior: "smooth" });
    }
  }, [location]);

  return (
    <div className="bg-light">
      {/* Add scroll-padding-top to offset fixed navbar */}
      <style>
        {`
          html {
            scroll-padding-top: 80px; /* adjust to your navbar height */
          }
        `}
      </style>

      {/* Hero Section */}
      <header className="py-5 text-center bg-white">
        <Container>
          <Row className="justify-content-center">
            <Col md={8}>
              <h1 className="display-5 fw-bold">{heroContent.title}</h1>
              <p className="lead text-muted my-4">{heroContent.subtitle}</p>
              <Button href="#how-it-works" size="lg" variant="primary">
                {heroContent.cta}
              </Button>
            </Col>
          </Row>
        </Container>
      </header>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-5">
        <Container>
          <div className="text-center mb-5">
            <h2 className="fw-bold">From Data to Discovery in 3 Steps</h2>
            <p className="text-muted">
              An intuitive workflow designed for clarity and speed.
            </p>
          </div>
          <Row className="g-4 text-center">
            {steps.map((step, idx) => (
              <Col md={4} key={idx}>
                <div className="p-4 bg-white rounded shadow-sm h-100">
                  <div className="text-primary mb-3 fs-1">
                    <i className={`bi ${step.icon}`}></i>
                  </div>
                  <h5 className="fw-semibold">{step.title}</h5>
                  <p className="text-muted">{step.description}</p>
                </div>
              </Col>
            ))}
          </Row>
        </Container>
      </section>

      {/* Features Section */}
      <section id="features" className="py-5 bg-white">
        <Container>
          <div className="text-center mb-5">
            <h2 className="fw-bold">Powerful Features for Everyone</h2>
            <p className="text-muted">
              Democratizing data analytics for your entire team.
            </p>
          </div>
          <Row>
            {features.slice(0, 3).map((f, idx) => (
              <FeatureCard key={idx} {...f} />
            ))}
          </Row>
          <Row className="mt-4 justify-content-center">
            {features.slice(3).map((f, idx) => (
              <FeatureCard key={idx} {...f} />
            ))}
          </Row>
        </Container>
      </section>

      {/* Call to Action */}
      <section className="py-5 text-center bg-light" id="about">
        <Container>
          <Row className="justify-content-center">
            <Col md={8}>
              <h2 className="fw-bold">{ctaContent.title}</h2>
              <p className="lead text-muted my-4">{ctaContent.subtitle}</p>
              <Button size="lg" variant="success" onClick={handleGetStarted}>
                {ctaContent.button}
              </Button>
            </Col>
          </Row>
        </Container>
      </section>
    </div>
  );
}
