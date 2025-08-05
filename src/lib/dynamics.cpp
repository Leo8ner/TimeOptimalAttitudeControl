#include <toac/dynamics.h>

using namespace casadi;

// Constructor implementation
ExplicitDynamics::ExplicitDynamics() {
    X = SX::vertcat({SX::sym("q", 4), SX::sym("w", 3)});
    U = SX::sym("tau", 3);
    dt = SX::sym("dt");

    SX q = X(Slice(0, 4));
    SX w = X(Slice(4, 7));

    SX S = skew4(w);
    SX q_dot = 0.5 * SX::mtimes(S, q);

    SX I = SX::diag(SX::vertcat({i_x, i_y, i_z}));
    SX I_inv = SX::diag(SX::vertcat({1.0/i_x, 1.0/i_y, 1.0/i_z}));
    SX w_dot = SX::mtimes(I_inv, (U - cross(w, SX::mtimes(I, w))));

    SX X_dot = SX::vertcat({q_dot, w_dot});
    SX X_next = rk4(X_dot, X, dt);
    F = Function("F", {X, U, dt}, {X_next});

}

// Skew-symmetric matrix
SX skew4(const SX& w) {
    SX S = SX::zeros(4, 4);
    S(0,1) = -w(0); S(0,2) = -w(1); S(0,3) = -w(2);
    S(1,0) =  w(0); S(1,2) =  w(2); S(1,3) = -w(1);
    S(2,0) =  w(1); S(2,1) = -w(2); S(2,3) =  w(0);
    S(3,0) =  w(2); S(3,1) =  w(1); S(3,2) = -w(0);
    return S;
}

// RK4 integrator
SX rk4(const SX& x_dot, const SX& x, const SX& dt) {
    SX k1{x_dot};
    SX k2{substitute(x_dot, x, x + dt / 2 * k1)};
    SX k3{substitute(x_dot, x, x + dt / 2 * k2)};
    SX k4{substitute(x_dot, x, x + dt * k3)};
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

ImplicitDynamics::ImplicitDynamics(const std::string& plugin) {
    SX X = SX::vertcat({SX::sym("q", 4), SX::sym("w", 3)});
    SX U = SX::sym("tau", 3);
    SX dt = SX::sym("dt");

    SX q = X(Slice(0, 4));
    SX w = X(Slice(4, 7));

    SX S = skew4(w);
    SX q_dot = 0.5 * SX::mtimes(S, q);

    SX I = SX::diag(SX::vertcat({i_x, i_y, i_z}));
    SX I_inv = SX::diag(SX::vertcat({1.0/i_x, 1.0/i_y, 1.0/i_z}));
    SX w_dot = SX::mtimes(I_inv, (U - cross(w, SX::mtimes(I, w))));

    SX X_dot = SX::vertcat({q_dot, w_dot});

    // Create integrator options
    SXDict dae = {{"x", X}, {"u", U}, {"p", dt}, {"ode", X_dot*dt}};
    Dict opts;
    if (plugin == "ipopt") {
        // opts["collocation_scheme"] = "legendre";
        // opts["interpolation_order"] = 3;
        // opts["simplify"] = true;
        // opts["rootfinder"] = "fast_newton";
        // Function f = integrator("f", "collocation", dae, opts);
        SX X_next = rk4(X_dot, X, dt);
        Function f = Function("F", {X, U, dt}, {X_next});
        F = f.map(n_stp, "unroll");

    } else if (plugin == "fatrop") {
        // opts["collocation_scheme"] = "radau";
        // opts["interpolation_order"] = 4;
        // opts["simplify"] = true;
        // opts["rootfinder"] = "fast_newton";
        // Function f = integrator("f", "rk", dae, opts);
        SX X_next = rk4(X_dot, X, dt);
        F = Function("F", {X, U, dt}, {X_next});
        //F = f;
    } else {
        throw std::invalid_argument("Unsupported solver type: " + plugin);
    }
}