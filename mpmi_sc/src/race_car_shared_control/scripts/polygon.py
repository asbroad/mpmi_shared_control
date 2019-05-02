
import math

def inflate(polys, dist):

  lines = []

  for idx in range(len(polys)):
    # get poly
    poly = polys[idx]
    if idx == len(polys) - 1:
      poly_next = polys[0]
    else:
      poly_next = polys[idx+1]
    # get two points defining line
    p1_l = poly[0]
    p1_r = poly[1]
    # get slope of line
    p1_s = (p1_l[1] - p1_r[1])/(p1_l[0] - p1_r[0])
    # get angle
    p1_a = math.atan(p1_s)
    # get sign
    p1_a_s = 1 if poly_next[0][1] > p1_l[1] else -1
    # move dist along line
    p1_l_n = [p1_l[0] + p1_a_s*math.cos(p1_a)*dist, p1_l[1] + p1_a_s*math.sin(p1_a)*dist]
    p1_r_n = [p1_r[0] - p1_a_s*math.cos(p1_a)*dist, p1_r[1] - p1_a_s*math.sin(p1_a)*dist]
    # store new values
    lines.append([p1_l_n, p1_r_n])

  inflated_polys = []
  for idx in range(len(lines)):
    if idx == 0:
      p1_l, p1_r = lines[-1]
    else:
      p1_l, p1_r = lines[idx - 1]
    p2_l, p2_r = lines[idx]
    inflated_polys.append([p1_l, p1_r, p2_r, p2_l])

  return inflated_polys
