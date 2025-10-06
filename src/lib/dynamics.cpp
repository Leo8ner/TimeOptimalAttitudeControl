#include <toac/dynamics.h>

using namespace casadi;

Dynamics::Dynamics(const std::string& plugin) {
    MX X = MX::vertcat({MX::sym("q", 4), MX::sym("w", 3)});
    MX U = MX::sym("tau", 3);
    MX dt = MX::sym("dt");

    MX q = X(Slice(0, 4));
    MX w = X(Slice(4, 7));

    MX S = skew4(w);
    MX q_dot = 0.5 * MX::mtimes(S, q);

    MX I = MX::diag(MX::vertcat({i_x, i_y, i_z}));
    MX I_inv = MX::diag(MX::vertcat({1.0/i_x, 1.0/i_y, 1.0/i_z}));
    MX w_dot = MX::mtimes(I_inv, (U - cross(w, MX::mtimes(I, w))));

    MX X_dot = MX::vertcat({q_dot, w_dot});

    // Create integrator options
    MXDict dae = {{"x", X}, {"u", U}, {"p", dt}, {"ode", X_dot*dt}};
    Dict opts;
    if (plugin == "ipopt" || plugin == "snopt") {
        // opts["collocation_scheme"] = "legendre";
        // opts["interpolation_order"] = 3;
        // opts["simplify"] = true;
        // opts["rootfinder"] = "fast_newton";
        // Function f = integrator("f", "collocation", dae, opts);
        MX X_next = rk4(X_dot, X, dt);
        Function f = Function("F", {X, U, dt}, {X_next});
        F = f.map(n_stp, "unroll");

    } else if (plugin == "fatrop") {
        // opts["collocation_scheme"] = "radau";
        // opts["interpolation_order"] = 4;
        // opts["simplify"] = true;
        // opts["rootfinder"] = "fast_newton";
        // Function f = integrator("f", "rk", dae, opts);
        MX X_next = rk4(X_dot, X, dt);
        F = Function("F", {X, U, dt}, {X_next});
    } else {
        throw std::invalid_argument("Unsupported solver type: " + plugin);
    }
}